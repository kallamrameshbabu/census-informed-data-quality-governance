"""
ztlf_corruption.py
------------------
Principled, reproducible defect injection with cell-level ground truth.

Why this exists
===============
The original ZTLF study injected exactly the defects its rules detected, so
precision and recall were 1.0 by construction. This module instead:

  1. Injects at *controlled contamination rates* (a sweep, not a single point),
     so detection performance becomes a curve that can rise or fall.
  2. Records a *cell-level ground-truth mask* for every corruption, enabling
     precision / recall / F1 against our quality gate AND against third-party
     baselines (Great Expectations, Soda Core, PyDeequ, Pandera).
  3. Implements missingness *mechanisms* (MCAR / MAR / MNAR), following the
     Jenga methodology (Schelter, Rukat & Biessmann, EDBT 2021), because
     detectors behave differently under each mechanism. Injecting only MCAR
     flatters the detector.
  4. Is fully deterministic: (seed, rate, dataset) -> identical corruption.

Design rule
===========
We inject only defect classes that the natural-defect census showed are ABSENT
or RARE in a given dataset. Injecting a class that already occurs naturally
would contaminate the ground truth, because we could not tell an injected
defect from a pre-existing one. This is why Phase 1 had to come first.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

ROW_ID = "_ztlf_row_id"


# ===========================================================================
# Ground-truth record
# ===========================================================================
def _mask_rows(row_ids, column, defect_class, corruption, before, after):
    return pd.DataFrame({
        "row_id": row_ids,
        "column": column,
        "defect_class": defect_class,
        "corruption": corruption,
        "value_before": pd.Series(before, dtype="object").astype(str).values,
        "value_after": pd.Series(after, dtype="object").astype(str).values,
    })



def _ensure_object(df: pd.DataFrame, col: str) -> None:
    """Widen a column to object dtype before writing mixed-type values.

    pandas 3.0 uses a dedicated `str` dtype that rejects non-string writes, so
    injecting a numeric or NaN into a text column raises TypeError. Corruption
    deliberately produces mixed types, so we widen first.
    """
    if col in df.columns and df[col].dtype != object:
        df[col] = df[col].astype(object)


def _select(rng: np.random.Generator, index: pd.Index, fraction: float) -> np.ndarray:
    """Choose a deterministic random subset of an index."""
    n = int(round(len(index) * fraction))
    if n <= 0:
        return np.array([], dtype=index.dtype)
    return rng.choice(np.asarray(index), size=min(n, len(index)), replace=False)


# ===========================================================================
# Corruption primitives
# ===========================================================================
@dataclass
class Corruption:
    """Base class. Subclasses implement `_apply`."""
    name: str
    defect_class: str
    columns: Sequence[str]

    def apply(self, df: pd.DataFrame, rng: np.random.Generator,
              fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
        raise NotImplementedError


@dataclass
class MissingnessCorruption(Corruption):
    """Blank out values under MCAR, MAR, or MNAR.

    MCAR : uniformly at random.
    MAR  : probability depends on ANOTHER observed column (`depends_on`).
    MNAR : probability depends on the value being hidden itself.
    """
    mechanism: str = "MCAR"
    sentinel: str = ""              # "" = true null; "?" = sentinel-encoded
    depends_on: str | None = None

    def apply(self, df, rng, fraction):
        out = df.copy()
        masks = []
        for col in self.columns:
            if col not in out.columns:
                continue
            if fraction <= 0:
                continue

            if self.mechanism == "MCAR":
                chosen = _select(rng, out.index, fraction)

            elif self.mechanism == "MAR":
                key = self.depends_on
                if key is None or key not in out.columns:
                    chosen = _select(rng, out.index, fraction)
                else:
                    # Rank by the *other* column; bias missingness toward the
                    # upper half of that ranking.
                    ranks = pd.to_numeric(out[key], errors="coerce").rank(
                        pct=True, method="first").fillna(0.5)
                    w = (0.2 + 1.6 * ranks).to_numpy()
                    chosen = _weighted_choice(rng, out.index, w, fraction)

            elif self.mechanism == "MNAR":
                ranks = pd.to_numeric(out[col], errors="coerce").rank(
                    pct=True, method="first")
                if ranks.isna().all():
                    # categorical: bias toward rarer categories
                    freq = out[col].astype(str).map(
                        out[col].astype(str).value_counts(normalize=True))
                    w = (1.0 / (freq + 1e-6)).to_numpy()
                    w = w / w.max()
                    w = 0.2 + 1.6 * w
                else:
                    w = (0.2 + 1.6 * ranks.fillna(0.5)).to_numpy()
                chosen = _weighted_choice(rng, out.index, w, fraction)
            else:
                raise ValueError(f"unknown mechanism {self.mechanism}")

            if len(chosen) == 0:
                continue
            before = out.loc[chosen, col].to_numpy()
            new_val = np.nan if self.sentinel == "" else self.sentinel
            _ensure_object(out, col)
            out.loc[chosen, col] = new_val
            masks.append(_mask_rows(out.loc[chosen, ROW_ID].to_numpy(), col,
                                    self.defect_class,
                                    f"{self.name}[{self.mechanism}]",
                                    before, [str(new_val)] * len(chosen)))
        return out, _concat(masks)


def _weighted_choice(rng, index, weights, fraction):
    n = int(round(len(index) * fraction))
    if n <= 0:
        return np.array([], dtype=index.dtype)
    p = np.clip(weights, 1e-9, None)
    p = p / p.sum()
    return rng.choice(np.asarray(index), size=min(n, len(index)),
                      replace=False, p=p)


@dataclass
class DomainCorruption(Corruption):
    """Replace a categorical value with a token outside the declared domain."""
    invalid_tokens: Sequence[str] = ("__OUT_OF_DOMAIN__",)

    def apply(self, df, rng, fraction):
        out = df.copy()
        masks = []
        for col in self.columns:
            if col not in out.columns or fraction <= 0:
                continue
            chosen = _select(rng, out.index, fraction)
            if len(chosen) == 0:
                continue
            before = out.loc[chosen, col].to_numpy()
            tokens = rng.choice(np.asarray(self.invalid_tokens), size=len(chosen))
            _ensure_object(out, col)
            out.loc[chosen, col] = tokens
            masks.append(_mask_rows(out.loc[chosen, ROW_ID].to_numpy(), col,
                                    self.defect_class, self.name,
                                    before, tokens))
        return out, _concat(masks)


@dataclass
class RangeCorruption(Corruption):
    """Push numeric values outside their plausible range.

    Modes: 'negative' (sign flip), 'extreme' (x1000), 'impossible' (fixed value).
    """
    mode: str = "extreme"
    impossible_value: float = 9_999_999.0

    def apply(self, df, rng, fraction):
        out = df.copy()
        masks = []
        for col in self.columns:
            if col not in out.columns or fraction <= 0:
                continue
            chosen = _select(rng, out.index, fraction)
            if len(chosen) == 0:
                continue
            before = out.loc[chosen, col].to_numpy()
            num = pd.to_numeric(pd.Series(before), errors="coerce").fillna(1.0)
            if self.mode == "negative":
                after = (-num.abs() - 1.0).to_numpy()
            elif self.mode == "extreme":
                after = (num.abs() * 1000.0 + 1000.0).to_numpy()
            else:
                after = np.full(len(chosen), self.impossible_value)
            _ensure_object(out, col)
            out.loc[chosen, col] = after
            masks.append(_mask_rows(out.loc[chosen, ROW_ID].to_numpy(), col,
                                    self.defect_class, f"{self.name}[{self.mode}]",
                                    before, after))
        return out, _concat(masks)


@dataclass
class RepresentationCorruption(Corruption):
    """Introduce case / whitespace / abbreviation variants of a valid value.

    These are REPAIRABLE defects: a good framework normalizes them rather than
    quarantining. Separating repairable from fatal is a ZTLF claim, so we must
    inject both kinds to test it.
    """
    abbreviations: Mapping[str, str] = field(default_factory=dict)

    def apply(self, df, rng, fraction):
        out = df.copy()
        masks = []
        variants = ["upper", "lower", "pad", "title", "abbrev"]
        for col in self.columns:
            if col not in out.columns or fraction <= 0:
                continue
            chosen = _select(rng, out.index, fraction)
            if len(chosen) == 0:
                continue
            before = out.loc[chosen, col].astype(str).to_numpy()
            kinds = rng.choice(variants, size=len(chosen))
            after = []
            for v, k in zip(before, kinds):
                if k == "upper":
                    after.append(v.upper())
                elif k == "lower":
                    after.append(v.lower())
                elif k == "pad":
                    after.append(f"  {v} ")
                elif k == "title":
                    after.append(v.title())
                else:
                    after.append(self.abbreviations.get(v, v.upper()[:1]))
            after = np.array(after, dtype=object)
            _ensure_object(out, col)
            out.loc[chosen, col] = after
            masks.append(_mask_rows(out.loc[chosen, ROW_ID].to_numpy(), col,
                                    self.defect_class, self.name, before, after))
        return out, _concat(masks)


@dataclass
class TypeCorruption(Corruption):
    """Write non-numeric text into a numeric column (structural / D1)."""
    tokens: Sequence[str] = ("N/A", "not recorded", "TBD", "#REF!")

    def apply(self, df, rng, fraction):
        out = df.copy()
        masks = []
        for col in self.columns:
            if col not in out.columns or fraction <= 0:
                continue
            chosen = _select(rng, out.index, fraction)
            if len(chosen) == 0:
                continue
            before = out.loc[chosen, col].to_numpy()
            toks = rng.choice(np.asarray(self.tokens), size=len(chosen))
            _ensure_object(out, col)
            out.loc[chosen, col] = toks
            masks.append(_mask_rows(out.loc[chosen, ROW_ID].to_numpy(), col,
                                    self.defect_class, self.name, before, toks))
        return out, _concat(masks)


@dataclass
class DuplicateRowCorruption(Corruption):
    """Append exact duplicate rows (row-level defect, not cell-level).

    The duplicate COPY is the defect; the original is legitimate. We mark the
    appended copies so scoring does not penalise a detector for keeping the
    original.
    """
    def apply(self, df, rng, fraction):
        if fraction <= 0 or len(df) == 0:
            return df.copy(), _concat([])
        chosen = _select(rng, df.index, fraction)
        if len(chosen) == 0:
            return df.copy(), _concat([])
        dupes = df.loc[chosen].copy()
        # New row ids for the copies, flagged so they are traceable
        dupes[ROW_ID] = [f"{r}__dup" for r in dupes[ROW_ID]]
        out = pd.concat([df, dupes], ignore_index=True)
        mask = _mask_rows(dupes[ROW_ID].to_numpy(), "__ROW__",
                          self.defect_class, self.name,
                          ["original"] * len(dupes), ["duplicate"] * len(dupes))
        return out, mask


@dataclass
class ConsistencyCorruption(Corruption):
    """Break a documented cross-field rule via a caller-supplied mutation."""
    mutate: Callable[[pd.DataFrame, np.ndarray], pd.DataFrame] = None

    def apply(self, df, rng, fraction):
        out = df.copy()
        if fraction <= 0 or self.mutate is None:
            return out, _concat([])
        chosen = _select(rng, out.index, fraction)
        if len(chosen) == 0:
            return out, _concat([])
        # pandas 3.0 uses a dedicated str dtype that rejects non-string writes.
        # Widen the targeted columns before handing control to the caller's
        # mutate(), which may legitimately write ints/floats.
        for c in self.columns:
            _ensure_object(out, c)
        before = out.loc[chosen, list(self.columns)].astype(str).agg("|".join, axis=1).to_numpy()
        out = self.mutate(out, chosen)
        after = out.loc[chosen, list(self.columns)].astype(str).agg("|".join, axis=1).to_numpy()
        mask = _mask_rows(out.loc[chosen, ROW_ID].to_numpy(),
                          "+".join(self.columns), self.defect_class,
                          self.name, before, after)
        return out, mask


def _concat(masks):
    cols = ["row_id", "column", "defect_class", "corruption",
            "value_before", "value_after"]
    masks = [m for m in masks if m is not None and len(m)]
    if not masks:
        return pd.DataFrame(columns=cols)
    return pd.concat(masks, ignore_index=True)


# ===========================================================================
# Experiment runner
# ===========================================================================
@dataclass
class CorruptionPlan:
    """An ordered set of corruptions applied at a shared contamination rate."""
    dataset: str
    corruptions: Sequence[Corruption]

    def run(self, df: pd.DataFrame, rate: float, seed: int
            ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if ROW_ID not in df.columns:
            raise ValueError(f"{ROW_ID} missing; assign stable row ids first")
        rng = np.random.default_rng(seed)
        work = df.copy()
        all_masks = []
        for c in self.corruptions:
            work, m = c.apply(work, rng, rate)
            if len(m):
                all_masks.append(m)

                # Duplicate rows are appended AFTER earlier corruptions, so the
                # copies carry any already-corrupted cell values under new row
                # ids. Those cells are genuinely defective and must appear in
                # ground truth, otherwise a detector that correctly flags them
                # is penalised as a false positive and precision is understated.
                if isinstance(c, DuplicateRowCorruption):
                    dup_ids = m.loc[m["column"] == "__ROW__", "row_id"].astype(str)
                    orig_of = {d: d[:-len("__dup")] for d in dup_ids
                               if d.endswith("__dup")}
                    if orig_of:
                        prior = pd.concat(all_masks[:-1], ignore_index=True) \
                            if len(all_masks) > 1 else None
                        if prior is not None and len(prior):
                            inherited = prior[
                                prior["row_id"].astype(str).isin(orig_of.values())
                            ].copy()
                            if len(inherited):
                                back = {v: k for k, v in orig_of.items()}
                                inherited["row_id"] = \
                                    inherited["row_id"].astype(str).map(back)
                                inherited["corruption"] = \
                                    inherited["corruption"].astype(str) + "+inherited_by_duplicate"
                                all_masks.append(inherited)

        mask = _concat(all_masks)
        if len(mask):
            mask.insert(0, "rate", rate)
            mask.insert(0, "seed", seed)
            mask.insert(0, "dataset", self.dataset)
        return work, mask


def assign_row_ids(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = df.copy()
    out.insert(0, ROW_ID, [f"{prefix}_{i:08d}" for i in range(len(out))])
    return out


def sweep(plan: CorruptionPlan, df: pd.DataFrame,
          rates: Sequence[float], seeds: Sequence[int]):
    """Yield (rate, seed, corrupted_df, ground_truth_mask) for every cell."""
    for rate in rates:
        for seed in seeds:
            corrupted, mask = plan.run(df, rate, seed)
            yield rate, seed, corrupted, mask


# ===========================================================================
# Scoring: detector output vs ground truth
# ===========================================================================
def score_detection(ground_truth: pd.DataFrame,
                    detected: pd.DataFrame,
                    level: str = "cell") -> dict:
    """Precision / recall / F1 of a detector against the injected ground truth.

    ground_truth : must have row_id, column
    detected     : must have row_id, column  (column='__ROW__' for row-level)
    level        : 'cell' -> match on (row_id, column)
                   'row'  -> match on row_id only
    """
    def keys(d):
        if len(d) == 0:
            return set()
        if level == "row":
            return set(d["row_id"].astype(str))
        return set(zip(d["row_id"].astype(str), d["column"].astype(str)))

    gt, dt = keys(ground_truth), keys(detected)
    tp = len(gt & dt)
    fp = len(dt - gt)
    fn = len(gt - dt)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"level": level, "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 6), "recall": round(recall, 6),
            "f1": round(f1, 6), "n_ground_truth": len(gt), "n_detected": len(dt)}


def score_by_defect_class(ground_truth: pd.DataFrame,
                          detected: pd.DataFrame,
                          level: str = "cell") -> pd.DataFrame:
    """Per-defect-class scoring done correctly.

    IMPORTANT: per-class *precision* is not well defined. A detection that
    correctly flags a D4 domain violation is not a false positive "for D1" --
    it simply belongs to another class. Computing precision per class against
    the full detection set therefore produces meaningless near-zero values.

    This function reports, per class:
      n_injected  : ground-truth cells of that class
      n_recovered : how many the detector found  (true positives)
      recall      : detection coverage for that class   <-- the meaningful metric
    and separately returns overall precision/recall/F1 across all classes,
    where a false positive is a detection matching no injected defect at all.
    """
    def keys(d):
        if len(d) == 0:
            return set()
        if level == "row":
            return set(d["row_id"].astype(str))
        return set(zip(d["row_id"].astype(str), d["column"].astype(str)))

    det_keys = keys(detected)
    all_gt_keys = keys(ground_truth)

    rows = []
    for cls in sorted(ground_truth["defect_class"].dropna().unique()):
        gt_cls = keys(ground_truth[ground_truth["defect_class"] == cls])
        tp = len(gt_cls & det_keys)
        rows.append({
            "defect_class": cls,
            "n_injected": len(gt_cls),
            "n_recovered": tp,
            "recall": round(tp / len(gt_cls), 6) if gt_cls else float("nan"),
        })

    tp_all = len(all_gt_keys & det_keys)
    fp_all = len(det_keys - all_gt_keys)
    fn_all = len(all_gt_keys - det_keys)
    precision = tp_all / (tp_all + fp_all) if (tp_all + fp_all) else 0.0
    recall = tp_all / (tp_all + fn_all) if (tp_all + fn_all) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)

    rows.append({
        "defect_class": "ALL",
        "n_injected": len(all_gt_keys),
        "n_recovered": tp_all,
        "recall": round(recall, 6),
    })
    out = pd.DataFrame(rows)
    out.attrs["overall"] = {
        "precision": round(precision, 6), "recall": round(recall, 6),
        "f1": round(f1, 6), "tp": tp_all, "fp": fp_all, "fn": fn_all,
    }
    return out


def natural_defect_keys(clean_df: pd.DataFrame,
                        columns: Sequence[str],
                        sentinels: Sequence[str] = ("?", "unknown", "NA",
                                                    "", " ", "None", "null"),
                        ) -> set:
    """(row_id, column) keys that were ALREADY defective before injection.

    Why this exists
    ---------------
    Scoring a detector against injected-only ground truth is biased whenever
    the source data contains natural defects. A detector that correctly flags
    a pre-existing '?' sentinel is counted as a false positive, because that
    cell was never injected. On the diabetes corpus this understates precision
    by roughly a factor of three.

    The correct evaluation restricts scoring to cells that were clean before
    injection, so every flag is attributable to the controlled experiment.
    Natural defects are reported separately via the Phase 1 census.
    """
    sent = {str(s).strip().lower() for s in sentinels}
    keys = set()
    ids = clean_df[ROW_ID].astype(str).to_numpy()
    for col in columns:
        if col not in clean_df.columns:
            continue
        s = clean_df[col]
        norm = s.astype(str).str.strip().str.lower()
        bad = s.isna().to_numpy() | norm.isin(sent).to_numpy()
        keys.update((r, col) for r in ids[bad])
    return keys


def score_restricted(ground_truth: pd.DataFrame,
                     detected: pd.DataFrame,
                     exclude_keys: set,
                     level: str = "cell") -> pd.DataFrame:
    """Per-class scoring with naturally-defective cells excluded.

    `exclude_keys` should come from natural_defect_keys() computed on the
    CLEAN frame. Detections and ground-truth entries on those cells are
    dropped before scoring, plus their duplicate-inherited copies.
    """
    def _drop(d):
        if len(d) == 0:
            return d
        rid = d["row_id"].astype(str)
        base = rid.str.replace("__dup$", "", regex=True)
        k_self = list(zip(rid, d["column"].astype(str)))
        k_base = list(zip(base, d["column"].astype(str)))
        keep = [(a not in exclude_keys) and (b not in exclude_keys)
                for a, b in zip(k_self, k_base)]
        return d.loc[keep]

    return score_by_defect_class(_drop(ground_truth), _drop(detected), level)


def natural_defect_keys_full(clean_df: pd.DataFrame,
                             spec,
                             gate_columns: Sequence[str],
                             extra_sentinels: Sequence[str] | None = None) -> set:
    """ALL naturally-defective (row_id, column) keys, not just missing ones.

    Why this supersedes natural_defect_keys()
    -----------------------------------------
    Excluding only sentinel/null cells leaves other pre-existing defect classes
    in play. On Online Retail II the corpus contains ~29k natural RANGE
    violations (negative Quantity/Price from returns and adjustments) and ~10k
    CONSISTENCY violations (cancellation semantics). A detector that correctly
    flags those is still scored as a false positive, so restricted precision
    sat at 0.52 instead of ~1.0 at low contamination.

    This function unions every naturally-defective cell the Phase 1 census can
    identify -- missingness, domain, and range -- restricted to the columns the
    detector actually inspects.

    `spec` is a ztlf_profiling.DatasetSpec supplying domains and ranges.
    """
    sentinels = tuple(extra_sentinels) if extra_sentinels is not None \
        else tuple(getattr(spec, "sentinels", ("?", "unknown", "NA", "", " ")))
    cols = [c for c in gate_columns if c in clean_df.columns]

    keys = natural_defect_keys(clean_df, cols, sentinels=sentinels)

    ids = clean_df[ROW_ID].astype(str).to_numpy()

    # Domain violations already present in the source
    for col, allowed in getattr(spec, "categorical_domains", {}).items():
        if col not in cols:
            continue
        s = clean_df[col].astype(str).str.strip()
        bad = (~s.isin({str(a) for a in allowed})).to_numpy()
        keys.update((r, col) for r in ids[bad])

    # Range violations already present in the source (negative prices, returns)
    for col, (lo, hi) in getattr(spec, "numeric_ranges", {}).items():
        if col not in cols:
            continue
        x = pd.to_numeric(clean_df[col], errors="coerce")
        bad = (x.isna() | (x < lo) | (x > hi)).to_numpy()
        keys.update((r, col) for r in ids[bad])

    return keys

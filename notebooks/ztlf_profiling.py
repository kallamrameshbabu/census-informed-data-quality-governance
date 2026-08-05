"""
ztlf_profiling.py
-----------------
Natural-defect census for the ZTLF rework.

Purpose
=======
The original ZTLF study injected synthetic defects and then detected exactly
those defects, producing a tautological result. This module instead measures
the *naturally occurring* defects already present in each source dataset,
before any injection. Those natural defects become ground-truth-adjacent
evidence that the quality gate detects real-world problems, not just planted
ones.

Defect taxonomy (dataset-independent):
  D1 STRUCTURAL      schema/parse failures, wrong column count, type coercion loss
  D2 MISSINGNESS     nulls and sentinel values ('?', 'unknown', 'NA', '', 'None')
  D3 UNIQUENESS      duplicate keys, fully duplicated rows
  D4 DOMAIN          values outside a declared categorical domain
  D5 RANGE           numeric values outside a declared plausible range
  D6 CONSISTENCY     cross-field contradictions (e.g. cancelled invoice, positive qty)
  D7 REPRESENTATION  case/whitespace/abbreviation variants of the same value

Design notes
============
* Pure pandas so it runs identically in Colab and locally; no Spark needed
  for profiling. Spark/Delta is used later for the medallion pipeline.
* Every function returns a tidy DataFrame so results can be written straight
  to CSV for the manuscript tables.
* Sentinel detection is explicit and configurable rather than hard-coded,
  because '?' is missingness in the diabetes data but could be legitimate
  text elsewhere.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype


def _is_texty(s: pd.Series) -> bool:
    """True for text-like columns under both pandas 2.x (object) and 3.x (str).

    pandas 3.0 introduced a dedicated string dtype, so an `== object` test
    silently skips every text column. This helper keeps the census correct
    across versions.
    """
    return s.dtype == object or is_string_dtype(s)

# Values that commonly encode missingness in public datasets but are not
# parsed as NaN by default. Kept explicit so the paper can report exactly
# what was treated as missing.
DEFAULT_SENTINELS: tuple[str, ...] = ("?", "unknown", "Unknown", "UNKNOWN",
                                      "NA", "N/A", "na", "none", "None",
                                      "null", "NULL", "", " ", "-")


@dataclass
class DatasetSpec:
    """Declarative description of a dataset for profiling.

    Only `name` and `path` are required; everything else refines the census.
    """
    name: str
    path: str
    read_kwargs: dict = field(default_factory=dict)
    key_columns: Sequence[str] = ()           # for D3 uniqueness
    sentinels: Sequence[str] = DEFAULT_SENTINELS
    categorical_domains: Mapping[str, set] = field(default_factory=dict)  # D4
    numeric_ranges: Mapping[str, tuple] = field(default_factory=dict)     # D5
    # D6 consistency rules: name -> callable(df) -> boolean Series (True = violation)
    consistency_rules: Mapping[str, object] = field(default_factory=dict)
    license_note: str = ""
    citation: str = ""


# --------------------------------------------------------------------------
# D2  Missingness (true nulls + sentinel values)
# --------------------------------------------------------------------------
def missingness_census(df: pd.DataFrame,
                       sentinels: Iterable[str] = DEFAULT_SENTINELS
                       ) -> pd.DataFrame:
    """Per-column count of true nulls and sentinel-encoded missing values."""
    sentinel_set = {str(s).strip().lower() for s in sentinels}
    rows = []
    n = len(df)
    for col in df.columns:
        s = df[col]
        true_null = int(s.isna().sum())
        if _is_texty(s):
            norm = s.astype(str).str.strip().str.lower()
            sentinel_hits = int(norm.isin(sentinel_set).sum() - s.isna().sum()
                                if s.isna().any() else norm.isin(sentinel_set).sum())
            sentinel_hits = max(sentinel_hits, 0)
            # which sentinel tokens actually appear, for the manuscript table
            present = sorted(set(norm[norm.isin(sentinel_set)].unique()))
        else:
            sentinel_hits = 0
            present = []
        total_missing = true_null + sentinel_hits
        rows.append({
            "column": col,
            "dtype": str(s.dtype),
            "n_rows": n,
            "true_null": true_null,
            "sentinel_missing": sentinel_hits,
            "total_missing": total_missing,
            "pct_missing": round(100.0 * total_missing / n, 4) if n else 0.0,
            "sentinel_tokens": ";".join(present),
        })
    out = pd.DataFrame(rows).sort_values("pct_missing", ascending=False)
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------
# D3  Uniqueness
# --------------------------------------------------------------------------
def uniqueness_census(df: pd.DataFrame,
                      key_columns: Sequence[str] = ()) -> pd.DataFrame:
    """Exact-duplicate rows and duplicate key occurrences."""
    rows = [{
        "check": "fully_duplicated_rows",
        "subject": "all_columns",
        "n_violations": int(df.duplicated(keep="first").sum()),
        "n_rows": len(df),
    }]
    for key in key_columns:
        if key in df.columns:
            rows.append({
                "check": "duplicate_key",
                "subject": key,
                "n_violations": int(df.duplicated(subset=[key], keep="first").sum()),
                "n_rows": len(df),
            })
    out = pd.DataFrame(rows)
    out["pct"] = (100.0 * out["n_violations"] / out["n_rows"]).round(4)
    return out


# --------------------------------------------------------------------------
# D4 / D5  Domain and range violations
# --------------------------------------------------------------------------
def domain_census(df: pd.DataFrame,
                  domains: Mapping[str, set]) -> pd.DataFrame:
    rows = []
    for col, allowed in domains.items():
        if col not in df.columns:
            continue
        s = df[col].astype(str).str.strip()
        bad = ~s.isin({str(a) for a in allowed})
        offending = sorted(s[bad].unique())[:10]
        rows.append({
            "check": "domain_violation",
            "column": col,
            "n_violations": int(bad.sum()),
            "n_rows": len(df),
            "pct": round(100.0 * bad.sum() / len(df), 4) if len(df) else 0.0,
            "example_values": ";".join(offending),
        })
    return pd.DataFrame(rows)


def range_census(df: pd.DataFrame,
                 ranges: Mapping[str, tuple]) -> pd.DataFrame:
    rows = []
    for col, (lo, hi) in ranges.items():
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        bad = (s < lo) | (s > hi)
        rows.append({
            "check": "range_violation",
            "column": col,
            "bound_low": lo,
            "bound_high": hi,
            "n_violations": int(bad.sum()),
            "n_rows": len(df),
            "pct": round(100.0 * bad.sum() / len(df), 4) if len(df) else 0.0,
            "observed_min": float(s.min()) if s.notna().any() else np.nan,
            "observed_max": float(s.max()) if s.notna().any() else np.nan,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# D7  Representation variants (case / whitespace / punctuation collisions)
# --------------------------------------------------------------------------
def representation_census(df: pd.DataFrame,
                          max_cardinality: int = 200) -> pd.DataFrame:
    """Detect categorical values that collapse to the same normalized form.

    A column with 'Male', 'male', ' MALE ' has 3 raw values but 1 normalized
    value -> 2 representation defects. Only applied to low-cardinality object
    columns, since free text would produce meaningless collisions.
    """
    rows = []
    for col in df.columns:
        s = df[col]
        if not _is_texty(s):
            continue
        raw_vals = s.dropna().astype(str).unique()
        if len(raw_vals) == 0 or len(raw_vals) > max_cardinality:
            continue
        norm_map: dict[str, set] = {}
        for v in raw_vals:
            k = re.sub(r"\s+", " ", v).strip().lower()
            norm_map.setdefault(k, set()).add(v)
        collisions = {k: v for k, v in norm_map.items() if len(v) > 1}
        if collisions:
            example = "; ".join(
                f"{k} <- {sorted(v)}" for k, v in list(collisions.items())[:3])
        else:
            example = ""
        rows.append({
            "check": "representation_variant",
            "column": col,
            "n_raw_values": len(raw_vals),
            "n_normalized_values": len(norm_map),
            "n_colliding_groups": len(collisions),
            "example_collisions": example,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# D6  Consistency (dataset-specific cross-field rules)
# --------------------------------------------------------------------------
def consistency_census(df: pd.DataFrame,
                       rules: Mapping[str, object]) -> pd.DataFrame:
    rows = []
    for name, fn in rules.items():
        mask = fn(df)
        rows.append({
            "check": "consistency_violation",
            "rule": name,
            "n_violations": int(pd.Series(mask).fillna(False).sum()),
            "n_rows": len(df),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["pct"] = (100.0 * out["n_violations"] / out["n_rows"]).round(4)
    return out


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def load_dataset(spec: DatasetSpec) -> pd.DataFrame:
    """Load with no automatic NA coercion so sentinels stay visible."""
    kw = dict(spec.read_kwargs)
    kw.setdefault("keep_default_na", False)
    kw.setdefault("na_values", [])
    kw.setdefault("dtype", str)     # read as text; typing happens explicitly later
    return pd.read_csv(spec.path, **kw)


def file_fingerprint(path: str) -> dict:
    """SHA-256 + size, so the manuscript can prove which file was used."""
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
            size += len(chunk)
    return {"sha256": h.hexdigest(), "bytes": size}


def profile_dataset(spec: DatasetSpec) -> dict[str, pd.DataFrame]:
    df = load_dataset(spec)
    fp = file_fingerprint(spec.path)

    overview = pd.DataFrame([{
        "dataset": spec.name,
        "n_rows": len(df),
        "n_columns": df.shape[1],
        "sha256": fp["sha256"],
        "bytes": fp["bytes"],
        "license": spec.license_note,
        "citation": spec.citation,
    }])

    results = {
        "overview": overview,
        "missingness": missingness_census(df, spec.sentinels),
        "uniqueness": uniqueness_census(df, spec.key_columns),
        "domain": domain_census(df, spec.categorical_domains),
        "range": range_census(df, spec.numeric_ranges),
        "representation": representation_census(df),
        "consistency": consistency_census(df, spec.consistency_rules),
    }
    for k, v in results.items():
        if k != "overview" and not v.empty and "dataset" not in v.columns:
            v.insert(0, "dataset", spec.name)
    return results


def summarize_census(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One-row-per-defect-class summary — this becomes a manuscript table."""
    name = results["overview"]["dataset"].iloc[0]
    n = int(results["overview"]["n_rows"].iloc[0])
    miss = results["missingness"]
    rows = [
        {"defect_class": "D2 Missingness",
         "n_affected_cells": int(miss["total_missing"].sum()),
         "n_affected_columns": int((miss["total_missing"] > 0).sum())},
        {"defect_class": "D3 Uniqueness",
         "n_affected_cells": int(results["uniqueness"]["n_violations"].sum()),
         "n_affected_columns": int((results["uniqueness"]["n_violations"] > 0).sum())},
        {"defect_class": "D4 Domain",
         "n_affected_cells": int(results["domain"]["n_violations"].sum()) if not results["domain"].empty else 0,
         "n_affected_columns": int((results["domain"]["n_violations"] > 0).sum()) if not results["domain"].empty else 0},
        {"defect_class": "D5 Range",
         "n_affected_cells": int(results["range"]["n_violations"].sum()) if not results["range"].empty else 0,
         "n_affected_columns": int((results["range"]["n_violations"] > 0).sum()) if not results["range"].empty else 0},
        {"defect_class": "D6 Consistency",
         "n_affected_cells": int(results["consistency"]["n_violations"].sum()) if not results["consistency"].empty else 0,
         "n_affected_columns": int((results["consistency"]["n_violations"] > 0).sum()) if not results["consistency"].empty else 0},
        {"defect_class": "D7 Representation",
         "n_affected_cells": int(results["representation"]["n_colliding_groups"].sum()) if not results["representation"].empty else 0,
         "n_affected_columns": int((results["representation"]["n_colliding_groups"] > 0).sum()) if not results["representation"].empty else 0},
    ]
    out = pd.DataFrame(rows)
    out.insert(0, "dataset", name)
    out.insert(2, "n_rows", n)
    return out

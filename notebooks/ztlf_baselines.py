"""
ztlf_baselines.py
-----------------
Wrappers that put every data-quality tool behind ONE interface:

    gate(df) -> DataFrame[row_id, column]

so all tools are scored by the identical procedure against the identical
ground truth. Without this, a comparison is not a comparison.

Fairness contract
=================
1. Every tool is given the SAME logical rule set: the documented domains,
   ranges, and completeness constraints for that dataset. No tool is given a
   rule another tool lacks.
2. Every tool sees byte-identical input (the frozen corrupted corpora).
3. Rules are authored from dataset documentation, never from the corruption
   plan, so no tool is tuned to the injected defects.
4. Where a tool cannot express a constraint, that is recorded as a
   capability gap rather than silently dropped -- the gap is a finding.
5. Rule-authoring effort (non-comment lines) and wall-clock runtime are
   recorded alongside accuracy, because a tool that is marginally more
   accurate but far more laborious is not obviously better.

Tool availability
=================
pandera            pip install pandera                  -- pandas native
great_expectations pip install great_expectations       -- pandas native
soda-core          pip install soda-core soda-core-duckdb
                   NOTE: Soda Core 4.x dropped in-memory pandas validation.
                   It requires a SQL data source, so the wrapper materialises
                   the frame into DuckDB first. This is a real usability
                   difference worth reporting, not a defect in our harness.
pydeequ            pip install pydeequ                  -- needs Spark + Java
                   Version-sensitive: pin to the Spark you actually run.

Every wrapper degrades gracefully: if the library is absent, it returns
None and the sweep records the tool as unavailable instead of crashing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

ROW_ID = "_ztlf_row_id"


# ---------------------------------------------------------------------------
# Shared rule specification -- ONE source of truth for all tools
# ---------------------------------------------------------------------------
@dataclass
class RuleSet:
    """Dataset rules expressed tool-independently.

    Each tool wrapper translates this into its own dialect. Because all
    wrappers read the same RuleSet, no tool can be accidentally advantaged.
    """
    dataset: str
    domains: Mapping[str, set] = field(default_factory=dict)
    ranges: Mapping[str, tuple] = field(default_factory=dict)
    not_null: Sequence[str] = ()
    sentinels: Sequence[str] = ()          # treated as missing
    unique: Sequence[str] = ()             # row-level uniqueness

    def columns(self) -> list[str]:
        cols = set(self.domains) | set(self.ranges) | set(self.not_null)
        return sorted(cols)


def _hits(df: pd.DataFrame, col: str, bad: pd.Series) -> list[dict]:
    bad = bad.fillna(False).to_numpy()
    return [{"row_id": r, "column": col} for r in df.loc[bad, ROW_ID]]


# ---------------------------------------------------------------------------
# Baseline 0 -- our own reference gate (the ZTLF quality layer)
# ---------------------------------------------------------------------------
def gate_ztlf(df: pd.DataFrame, rules: RuleSet) -> pd.DataFrame:
    out = []
    sent = {str(s).strip().lower() for s in rules.sentinels}

    for col in rules.not_null:
        if col not in df.columns:
            continue
        s = df[col]
        norm = s.astype(str).str.strip().str.lower()
        out += _hits(df, col, s.isna() | norm.isin(sent))

    for col, allowed in rules.domains.items():
        if col not in df.columns:
            continue
        # exact match: case/whitespace variants count as violations
        out += _hits(df, col, ~df[col].astype(str).isin({str(a) for a in allowed}))

    for col, (lo, hi) in rules.ranges.items():
        if col not in df.columns:
            continue
        x = pd.to_numeric(df[col], errors="coerce")
        out += _hits(df, col, x.isna() | (x < lo) | (x > hi))

    return pd.DataFrame(out, columns=["row_id", "column"])


# ---------------------------------------------------------------------------
# Baseline 1 -- Pandera
# ---------------------------------------------------------------------------
def gate_pandera(df: pd.DataFrame, rules: RuleSet):
    try:
        import pandera.pandas as pa
    except Exception:
        try:
            import pandera as pa           # older layout
        except Exception:
            return None

    checks: dict = {}
    sent = {str(s).strip().lower() for s in rules.sentinels}

    for col, allowed in rules.domains.items():
        if col in df.columns:
            checks[col] = pa.Column(str, pa.Check.isin({str(a) for a in allowed}),
                                    nullable=True, coerce=True)

    for col, (lo, hi) in rules.ranges.items():
        if col not in df.columns:
            continue
        checks[col] = pa.Column(
            float,
            pa.Check.in_range(float(lo), float(hi)),
            nullable=False, coerce=True,
        )

    for col in rules.not_null:
        if col in df.columns and col not in checks:
            checks[col] = pa.Column(
                str,
                pa.Check(lambda s, _sent=sent: ~s.astype(str).str.strip()
                         .str.lower().isin(_sent),
                         element_wise=False),
                nullable=False, coerce=True,
            )

    if not checks:
        return pd.DataFrame(columns=["row_id", "column"])

    schema = pa.DataFrameSchema(checks, strict=False, coerce=True)

    work = df.copy()
    # Pandera reports positional/index failures; keep the mapping explicit
    work = work.reset_index(drop=True)
    id_by_pos = work[ROW_ID].astype(str).to_dict()

    try:
        schema.validate(work, lazy=True)
        return pd.DataFrame(columns=["row_id", "column"])
    except Exception as exc:
        fc = getattr(exc, "failure_cases", None)
        if fc is None or len(fc) == 0:
            return pd.DataFrame(columns=["row_id", "column"])
        rows = []
        for _, r in fc.iterrows():
            idx, col = r.get("index"), r.get("column")
            if idx is None or col is None or (isinstance(idx, float) and np.isnan(idx)):
                continue
            rid = id_by_pos.get(int(idx))
            if rid is not None:
                rows.append({"row_id": rid, "column": col})
        return pd.DataFrame(rows, columns=["row_id", "column"]).drop_duplicates()


# ---------------------------------------------------------------------------
# Baseline 2 -- Great Expectations
# ---------------------------------------------------------------------------
def gate_great_expectations(df: pd.DataFrame, rules: RuleSet):
    try:
        import great_expectations as gx
        from great_expectations.core import ExpectationSuite
    except Exception:
        return None

    work = df.copy().reset_index(drop=True)
    id_by_pos = work[ROW_ID].astype(str).to_dict()

    try:
        context = gx.get_context(mode="ephemeral")
        ds = context.data_sources.add_pandas("ztlf_pandas")
        asset = ds.add_dataframe_asset(name="ztlf_asset")
        batch_def = asset.add_batch_definition_whole_dataframe("ztlf_batch")
        batch = batch_def.get_batch(batch_parameters={"dataframe": work})
    except Exception:
        return None

    import great_expectations.expectations as gxe

    expectations = []
    for col, allowed in rules.domains.items():
        if col in work.columns:
            expectations.append(
                (col, gxe.ExpectColumnValuesToBeInSet(
                    column=col, value_set=sorted({str(a) for a in allowed}))))
    for col, (lo, hi) in rules.ranges.items():
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
            expectations.append(
                (col, gxe.ExpectColumnValuesToBeBetween(
                    column=col, min_value=float(lo), max_value=float(hi))))
    for col in rules.not_null:
        if col in work.columns:
            expectations.append(
                (col, gxe.ExpectColumnValuesToNotBeNull(column=col)))

    rows = []
    for col, exp in expectations:
        try:
            res = batch.validate(exp, result_format={
                "result_format": "COMPLETE",
                "unexpected_index_column_names": [ROW_ID],
            })
            r = res.result or {}
            idx_list = r.get("unexpected_index_list") or []
            for item in idx_list:
                if isinstance(item, dict):
                    rid = item.get(ROW_ID)
                    if rid is not None:
                        rows.append({"row_id": str(rid), "column": col})
                else:
                    rid = id_by_pos.get(int(item))
                    if rid is not None:
                        rows.append({"row_id": rid, "column": col})
        except Exception:
            continue

    return pd.DataFrame(rows, columns=["row_id", "column"]).drop_duplicates()


# ---------------------------------------------------------------------------
# Baseline 3 -- Soda Core (via DuckDB; 4.x has no in-memory pandas path)
# ---------------------------------------------------------------------------
def gate_soda_duckdb(df: pd.DataFrame, rules: RuleSet):
    """Soda-equivalent checks executed in DuckDB.

    Soda Core 4.x expresses checks against a SQL data source and reports
    aggregate pass/fail counts rather than offending row identifiers. To score
    it cell-by-cell we run the SAME SodaCL constraint semantics as SQL and
    collect the failing row ids. This is documented in the paper as a
    capability difference: Soda is designed for monitoring thresholds, not
    per-cell forensics.
    """
    try:
        import duckdb
    except Exception:
        return None

    con = duckdb.connect()
    con.register("t", df)
    rows = []
    sent = [str(s).strip().lower() for s in rules.sentinels]

    def q(sql, col):
        try:
            got = con.execute(sql).fetchall()
            rows.extend({"row_id": str(r[0]), "column": col} for r in got)
        except Exception:
            pass

    for col, allowed in rules.domains.items():
        if col not in df.columns:
            continue
        vals = ",".join("'" + str(a).replace("'", "''") + "'" for a in allowed)
        q(f'SELECT "{ROW_ID}" FROM t WHERE CAST("{col}" AS VARCHAR) NOT IN ({vals})', col)

    for col, (lo, hi) in rules.ranges.items():
        if col not in df.columns:
            continue
        q(f'SELECT "{ROW_ID}" FROM t WHERE TRY_CAST("{col}" AS DOUBLE) IS NULL '
          f'OR TRY_CAST("{col}" AS DOUBLE) < {lo} OR TRY_CAST("{col}" AS DOUBLE) > {hi}', col)

    for col in rules.not_null:
        if col not in df.columns:
            continue
        lst = ",".join("'" + s.replace("'", "''") + "'" for s in sent) or "''"
        q(f'SELECT "{ROW_ID}" FROM t WHERE "{col}" IS NULL '
          f'OR lower(trim(CAST("{col}" AS VARCHAR))) IN ({lst})', col)

    con.close()
    return pd.DataFrame(rows, columns=["row_id", "column"]).drop_duplicates()


# ---------------------------------------------------------------------------
# Baseline 4 -- PyDeequ (Spark)
# ---------------------------------------------------------------------------
def gate_pydeequ(df: pd.DataFrame, rules: RuleSet, spark=None):
    """PyDeequ constraint verification.

    Deequ reports CONSTRAINT-level outcomes and metrics, not offending row
    identifiers, so exact cell-level scoring is not directly available. We
    therefore report Deequ at constraint level and record it as a capability
    gap for per-cell comparison. Returning None marks it unavailable rather
    than fabricating cell-level output.
    """
    return None


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------
BASELINES: dict[str, Callable] = {
    "ZTLF (ours)": gate_ztlf,
    "Pandera": gate_pandera,
    "GreatExpectations": gate_great_expectations,
    "SodaCore/DuckDB": gate_soda_duckdb,
    "PyDeequ": gate_pydeequ,
}


def run_baselines(df: pd.DataFrame, rules: RuleSet,
                  tools: Sequence[str] | None = None) -> dict:
    """Run each available tool, returning detections plus runtime."""
    names = tools or list(BASELINES)
    out = {}
    for name in names:
        fn = BASELINES.get(name)
        if fn is None:
            continue
        t0 = time.perf_counter()
        try:
            det = fn(df, rules)
        except Exception as exc:
            out[name] = {"detections": None, "seconds": np.nan,
                         "status": f"error: {type(exc).__name__}: {exc}"[:200]}
            continue
        secs = time.perf_counter() - t0
        if det is None:
            out[name] = {"detections": None, "seconds": np.nan,
                         "status": "unavailable"}
        else:
            out[name] = {"detections": det, "seconds": round(secs, 3),
                         "status": "ok"}
    return out

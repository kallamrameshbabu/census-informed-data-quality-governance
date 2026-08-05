"""
ztlf_downstream.py
------------------
Does passing the quality gate actually improve downstream model performance?

This module finally puts evidence behind the original paper's "AI-ready"
claim, which was previously asserted and never tested.

Experimental protocol (CleanML-style)
=====================================
The single most important design decision: **corruption is applied to the
TRAINING set only, and every condition is evaluated on the same clean, held-out
test set.**

Why this matters. If you corrupt the whole corpus and then split, the test set
is dirty too, and quarantine "improves" scores partly by removing hard test
rows. That is a leakage artifact, not a data-quality effect. Holding the test
set clean and fixed makes the comparison a genuine measurement of how training
data quality propagates to model behaviour.

Conditions compared (same model, same hyperparameters, same test set):
  clean        train on uncorrupted training data          (upper bound)
  corrupted    train on corrupted data, no quality gate    (lower bound)
  quarantine   drop every row the gate flags               (ZTLF default)
  repair       normalise representation, impute the rest   (alternative policy)

Reported per condition:
  roc_auc, pr_auc   discrimination
  brier             calibration -- lower is better; often degrades even when
                    AUC looks stable, which is why AUC alone is insufficient
  n_train           rows surviving the policy (the cost of quarantine)
  subgroup recall   per protected group, to expose disparate data loss

Subgroup analysis
=================
Quarantine deletes rows. If defects are not uniformly distributed across
groups, quarantine deletes groups unevenly. A policy can raise aggregate AUC
while degrading recall for a minority subgroup. That trade-off is invisible in
aggregate metrics and is exactly what a governance framework should surface.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROW_ID = "_ztlf_row_id"


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------
@dataclass
class DownstreamTask:
    """A supervised task defined over one corpus."""
    dataset: str
    target: str
    positive_label: Callable[[pd.Series], pd.Series]
    feature_columns: Sequence[str]
    numeric_columns: Sequence[str]
    categorical_columns: Sequence[str]
    subgroup_columns: Sequence[str] = ()
    drop_columns: Sequence[str] = ()


def bank_task() -> DownstreamTask:
    num = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
    cat = ["job", "marital", "education", "default", "housing", "loan",
           "contact", "month", "poutcome"]
    return DownstreamTask(
        dataset="bank_marketing_full",
        target="y",
        positive_label=lambda s: s.astype(str).str.strip().str.lower().eq("yes"),
        feature_columns=num + cat,
        numeric_columns=num,
        categorical_columns=cat,
        # marital and age band are the standard protected proxies for this corpus
        subgroup_columns=["marital", "age_band"],
    )


def diabetes_task() -> DownstreamTask:
    num = ["time_in_hospital", "num_lab_procedures", "num_procedures",
           "num_medications", "number_outpatient", "number_emergency",
           "number_inpatient", "number_diagnoses"]
    cat = ["race", "gender", "age", "admission_type_id",
           "discharge_disposition_id", "admission_source_id",
           "max_glu_serum", "A1Cresult", "metformin", "insulin",
           "change", "diabetesMed"]
    return DownstreamTask(
        dataset="diabetes_130us",
        target="readmitted",
        # clinically meaningful binarisation: early readmission vs not
        positive_label=lambda s: s.astype(str).str.strip().eq("<30"),
        feature_columns=num + cat,
        numeric_columns=num,
        categorical_columns=cat,
        subgroup_columns=["race", "gender"],
    )


TASKS = {"bank_marketing_full": bank_task, "diabetes_130us": diabetes_task}


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------
def add_derived_subgroups(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "age" in out.columns and "age_band" not in out.columns:
        a = pd.to_numeric(out["age"], errors="coerce")
        if a.notna().mean() > 0.5:        # numeric age (bank), not buckets
            out["age_band"] = pd.cut(a, [0, 30, 45, 60, 200],
                                     labels=["<30", "30-44", "45-59", "60+"]) \
                                .astype(str)
    return out


def build_matrix(train: pd.DataFrame, test: pd.DataFrame, task: DownstreamTask):
    """One-hot encode categoricals fitted on TRAIN only (no test leakage)."""
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.impute import SimpleImputer

    num_cols = [c for c in task.numeric_columns if c in train.columns]
    cat_cols = [c for c in task.categorical_columns if c in train.columns]

    def num_block(d):
        x = d[num_cols].apply(pd.to_numeric, errors="coerce")
        return x.to_numpy(dtype=float)

    imp = SimpleImputer(strategy="median")
    Xn_tr = imp.fit_transform(num_block(train))
    Xn_te = imp.transform(num_block(test))

    enc = OneHotEncoder(handle_unknown="ignore", min_frequency=20,
                        sparse_output=False)
    Xc_tr = enc.fit_transform(train[cat_cols].astype(str))
    Xc_te = enc.transform(test[cat_cols].astype(str))

    return (np.hstack([Xn_tr, Xc_tr]), np.hstack([Xn_te, Xc_te]))


# ---------------------------------------------------------------------------
# Quality policies
# ---------------------------------------------------------------------------
def policy_quarantine(train: pd.DataFrame, detections: pd.DataFrame) -> pd.DataFrame:
    """Drop every row containing at least one flagged cell (ZTLF default)."""
    if detections is None or len(detections) == 0:
        return train
    bad = set(detections["row_id"].astype(str))
    return train[~train[ROW_ID].astype(str).isin(bad)]


def policy_repair(train: pd.DataFrame, detections: pd.DataFrame,
                  task: DownstreamTask) -> pd.DataFrame:
    """Normalise representation and null out irreparable cells.

    Repair keeps the row and blanks only the offending cell, so the imputer
    handles it downstream. This trades data volume for value uncertainty --
    the opposite trade-off to quarantine.
    """
    out = train.copy()
    for c in task.categorical_columns:
        if c in out.columns:
            out[c] = out[c].astype(str).str.strip().str.lower()
    if detections is not None and len(detections):
        for col, grp in detections.groupby("column"):
            if col not in out.columns:
                continue
            ids = set(grp["row_id"].astype(str))
            m = out[ROW_ID].astype(str).isin(ids)
            out.loc[m, col] = np.nan
    return out


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def _fit_predict(Xtr, ytr, Xte, model_name: str, seed: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier

    if model_name == "logreg":
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        Xtr, Xte = sc.fit_transform(Xtr), sc.transform(Xte)
        m = LogisticRegression(max_iter=2000, random_state=seed)
    else:
        m = HistGradientBoostingClassifier(random_state=seed, max_iter=200)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def evaluate_condition(train: pd.DataFrame, test: pd.DataFrame,
                       task: DownstreamTask, condition: str,
                       model_name: str, seed: int) -> dict:
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                 brier_score_loss)

    ytr = task.positive_label(train[task.target]).to_numpy().astype(int)
    yte = task.positive_label(test[task.target]).to_numpy().astype(int)

    if len(np.unique(ytr)) < 2 or len(train) < 100:
        return {"condition": condition, "model": model_name, "seed": seed,
                "roc_auc": np.nan, "pr_auc": np.nan, "brier": np.nan,
                "n_train": len(train), "status": "degenerate"}

    Xtr, Xte = build_matrix(train, test, task)
    p = _fit_predict(Xtr, ytr, Xte, model_name, seed)

    return {"condition": condition, "model": model_name, "seed": seed,
            "roc_auc": roc_auc_score(yte, p),
            "pr_auc": average_precision_score(yte, p),
            "brier": brier_score_loss(yte, p),
            "n_train": len(train), "status": "ok",
            "_pred": p, "_ytrue": yte}


def subgroup_metrics(test: pd.DataFrame, y_true: np.ndarray, p: np.ndarray,
                     subgroup_columns: Sequence[str],
                     threshold: float | None = None) -> pd.DataFrame:
    """Per-group recall and positive rate.

    Threshold selection matters more than it looks. With an ~11% base rate
    (diabetes readmitted<30), a fixed 0.5 cut-off makes the model predict
    almost no positives, so recall collapses to ~0.01 for every subgroup and
    the comparison carries no information. Defaulting to a PREVALENCE-MATCHED
    threshold -- predict positive for the top q of scores, where q is the
    observed base rate -- keeps the operating point comparable across
    conditions and datasets.
    """
    from sklearn.metrics import recall_score
    rows = []
    if threshold is None:
        base = float(np.mean(y_true))
        threshold = float(np.quantile(p, 1.0 - base)) if 0 < base < 1 else 0.5
    yhat = (p >= threshold).astype(int)
    for col in subgroup_columns:
        if col not in test.columns:
            continue
        groups = test[col].astype(str).to_numpy()
        for g in pd.unique(groups):
            m = groups == g
            if m.sum() < 50:
                continue
            rows.append({
                "subgroup_column": col, "subgroup": g, "n_test": int(m.sum()),
                "recall": recall_score(y_true[m], yhat[m], zero_division=0),
                "positive_rate": float(yhat[m].mean()),
                "base_rate": float(y_true[m].mean()),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Census-informed rule authoring
# ---------------------------------------------------------------------------
def census_informed_not_null(clean_df, candidate_columns, sentinels=("?",),
                             max_natural_missing_pct: float = 5.0):
    """Keep only columns whose NATURAL missingness is below a threshold.

    Applying a not-null rule to a column that is missing by design is the
    single most destructive rule-authoring mistake available. On the diabetes
    corpus, `weight` is 96.86% missing by documented design; a not-null rule
    on it flags nearly every row, and row-level quarantine then deletes the
    corpus while the framework still reports 100% rule compliance.

    Returns (kept, rejected_with_pct) so the paper can report exactly which
    rules the census removed and why.
    """
    sent = {str(x).strip().lower() for x in sentinels}
    kept, rejected = [], {}
    for c in candidate_columns:
        if c not in clean_df.columns:
            continue
        s = clean_df[c]
        norm = s.astype(str).str.strip().str.lower()
        pct = 100.0 * float((s.isna() | norm.isin(sent)).mean())
        if pct <= max_natural_missing_pct:
            kept.append(c)
        else:
            rejected[c] = round(pct, 2)
    return kept, rejected
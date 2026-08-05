"""
ztlf_plans.py
-------------
Dataset-specific corruption plans.

Selection principle
===================
We inject a defect class into a dataset ONLY where the Phase 1 natural-defect
census showed that class is absent or negligible. Injecting a class that
already occurs naturally would make ground truth ambiguous: we could not tell
an injected defect from a pre-existing one.

Census result that drives these choices (Phase 1, measured):

  bank_marketing_full   D2 present (52,124 sentinel cells) -> DO NOT inject D2
                        D3/D4/D5/D6/D7 absent               -> safe to inject
  diabetes_130us        D2 present (192,849 cells)          -> DO NOT inject D2
                        D3 present (30,248 dup patient_nbr) -> DO NOT inject D3
                        D6 present (19,308 rows)            -> DO NOT inject D6
                        D1/D4/D5/D7 absent                  -> safe to inject
  online_retail_ii      D6 present (cancellations)          -> DO NOT inject D6
                        confirm remaining classes from your census run

This is a defensible, reportable justification for the injection design --
unlike the original study, where the seven injected categories were arbitrary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ztlf_corruption import (
    ConsistencyCorruption, CorruptionPlan, DomainCorruption,
    DuplicateRowCorruption, MissingnessCorruption, RangeCorruption,
    RepresentationCorruption, TypeCorruption,
)

OUT_OF_DOMAIN = ("__INVALID__", "n/a", "XXX", "999")


# ---------------------------------------------------------------------------
# Bank Marketing
# ---------------------------------------------------------------------------
def bank_plan(dataset_name: str) -> CorruptionPlan:
    def break_pdays_rule(df, idx):
        out = df.copy()
        out.loc[idx, "pdays"] = -1
        out.loc[idx, "previous"] = 7          # contradicts pdays == -1
        return out

    return CorruptionPlan(
        dataset=dataset_name,
        corruptions=[
            DomainCorruption("domain_job_marital", "D4",
                             ["job", "marital"], invalid_tokens=OUT_OF_DOMAIN),
            RangeCorruption("range_age_extreme", "D5", ["age"], mode="extreme"),
            RangeCorruption("range_duration_negative", "D5",
                            ["duration"], mode="negative"),
            RepresentationCorruption("repr_education_month", "D7",
                                     ["education", "month"],
                                     abbreviations={"married": "M",
                                                    "single": "S"}),
            TypeCorruption("type_balance", "D1", ["balance"]),
            DuplicateRowCorruption("duplicate_rows", "D3", []),
            ConsistencyCorruption("consistency_pdays", "D6",
                                  ["pdays", "previous"],
                                  mutate=break_pdays_rule),
        ],
    )


# ---------------------------------------------------------------------------
# Diabetes 130-US
#   No D2 (already 96.9% missing in weight), no D3 (patient_nbr repeats by
#   design), no D6 (change/drug contradictions already present).
# ---------------------------------------------------------------------------
def diabetes_plan() -> CorruptionPlan:
    return CorruptionPlan(
        dataset="diabetes_130us",
        corruptions=[
            DomainCorruption("domain_gender_readmit", "D4",
                             ["gender", "readmitted"],
                             invalid_tokens=OUT_OF_DOMAIN),
            DomainCorruption("domain_drug_levels", "D4",
                             ["metformin", "insulin"],
                             invalid_tokens=OUT_OF_DOMAIN),
            RangeCorruption("range_time_in_hospital", "D5",
                            ["time_in_hospital"], mode="extreme"),
            RangeCorruption("range_num_medications_neg", "D5",
                            ["num_medications"], mode="negative"),
            RepresentationCorruption("repr_race_agebucket", "D7",
                                     ["race", "age"]),
            TypeCorruption("type_num_lab_procedures", "D1",
                           ["num_lab_procedures"]),
        ],
    )


# ---------------------------------------------------------------------------
# Online Retail II
#   Cancellations already provide D6 naturally; negative quantities are
#   legitimate returns, so we do NOT inject sign flips on Quantity.
# ---------------------------------------------------------------------------
def online_retail_plan() -> CorruptionPlan:
    return CorruptionPlan(
        dataset="online_retail_ii",
        corruptions=[
            MissingnessCorruption("missing_description_MNAR", "D2",
                                  ["Description"], mechanism="MNAR",
                                  sentinel=""),
            MissingnessCorruption("missing_customer_MAR", "D2",
                                  ["Customer ID"], mechanism="MAR",
                                  sentinel="", depends_on="Price"),
            RangeCorruption("range_price_extreme", "D5", ["Price"],
                            mode="extreme"),
            RepresentationCorruption("repr_country", "D7", ["Country"]),
            TypeCorruption("type_quantity", "D1", ["Quantity"]),
            DuplicateRowCorruption("duplicate_rows", "D3", []),
        ],
    )


PLANS = {
    "bank_marketing_small": lambda: bank_plan("bank_marketing_small"),
    "bank_marketing_full": lambda: bank_plan("bank_marketing_full"),
    "diabetes_130us": diabetes_plan,
    "online_retail_ii": online_retail_plan,
}

# Experiment grid -----------------------------------------------------------
CONTAMINATION_RATES = (0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30)
N_SEEDS = 10


def seed_list(base_seed: int, n: int = N_SEEDS) -> list[int]:
    """Deterministic, reportable seed list derived from the project seed."""
    return [base_seed + i for i in range(n)]

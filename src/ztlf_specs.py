"""
ztlf_specs.py
-------------
Declarative specifications for the three ZTLF evaluation datasets.

Domains and ranges are derived from each dataset's official documentation,
not invented, so the quality rules are defensible to a referee.
"""

import pandas as pd

from ztlf_profiling import DatasetSpec, DEFAULT_SENTINELS

# ---------------------------------------------------------------------------
# 1. UCI Bank Marketing  (semicolon-delimited, quoted)
#    Moro, Cortez & Rita (2014), Decision Support Systems 62:22-31
#    CC BY 4.0
# ---------------------------------------------------------------------------
BANK_JOBS = {"admin.", "unknown", "unemployed", "management", "housemaid",
             "entrepreneur", "student", "blue-collar", "self-employed",
             "retired", "technician", "services"}
BANK_MARITAL = {"married", "divorced", "single"}
BANK_EDUCATION = {"unknown", "secondary", "primary", "tertiary"}
BANK_BINARY = {"yes", "no"}
BANK_CONTACT = {"unknown", "telephone", "cellular"}
BANK_MONTH = {"jan", "feb", "mar", "apr", "may", "jun",
              "jul", "aug", "sep", "oct", "nov", "dec"}
BANK_POUTCOME = {"unknown", "other", "failure", "success"}


def bank_spec(name: str, path: str) -> DatasetSpec:
    return DatasetSpec(
        name=name,
        path=path,
        read_kwargs={"sep": ";", "quotechar": '"'},
        key_columns=(),                      # no natural key: important finding
        sentinels=DEFAULT_SENTINELS,
        categorical_domains={
            "job": BANK_JOBS, "marital": BANK_MARITAL,
            "education": BANK_EDUCATION, "default": BANK_BINARY,
            "housing": BANK_BINARY, "loan": BANK_BINARY,
            "contact": BANK_CONTACT, "month": BANK_MONTH,
            "poutcome": BANK_POUTCOME, "y": BANK_BINARY,
        },
        numeric_ranges={
            "age": (18, 100),
            "day": (1, 31),
            "duration": (0, 5000),
            "campaign": (1, 100),
            "previous": (0, 300),
            "balance": (-10_000, 200_000),
        },
        consistency_rules={
            # pdays == -1 means 'not previously contacted'; if so, previous
            # must be 0 and poutcome must be 'unknown'. A real cross-field rule.
            "pdays_neg1_implies_previous_zero":
                lambda d: (pd.to_numeric(d["pdays"], errors="coerce") == -1)
                          & (pd.to_numeric(d["previous"], errors="coerce") != 0),
            "pdays_neg1_implies_poutcome_unknown":
                lambda d: (pd.to_numeric(d["pdays"], errors="coerce") == -1)
                          & (d["poutcome"].str.strip().str.lower() != "unknown"),
            "duration_zero_but_contacted":
                lambda d: (pd.to_numeric(d["duration"], errors="coerce") == 0)
                          & (d["y"].str.strip().str.lower() == "yes"),
        },
        license_note="CC BY 4.0 (UCI ML Repository, dataset 222)",
        citation=("Moro S, Cortez P, Rita P. A data-driven approach to predict "
                  "the success of bank telemarketing. Decision Support Systems. "
                  "2014;62:22-31. DOI 10.24432/C5K306"),
    )


# ---------------------------------------------------------------------------
# 2. UCI Diabetes 130-US Hospitals 1999-2008
#    Strack et al. (2014), BioMed Research International, art. 781670
#    CC BY 4.0 -- '?' is the documented missing-value sentinel
# ---------------------------------------------------------------------------
DIAB_GENDER = {"Male", "Female", "Unknown/Invalid"}
DIAB_AGE_BUCKETS = {f"[{i}-{i+10})" for i in range(0, 100, 10)}
DIAB_DRUG_LEVELS = {"No", "Steady", "Up", "Down"}
DIAB_READMIT = {"NO", "<30", ">30"}
DIAB_DRUG_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
]

DIABETES_SPEC = DatasetSpec(
    name="diabetes_130us",
    path="/mnt/project/diabetic_data.csv",
    read_kwargs={"sep": ","},
    key_columns=("encounter_id", "patient_nbr"),
    sentinels=("?",),          # documented sentinel; 'None'/'No' are meaningful here
    categorical_domains={
        "gender": DIAB_GENDER,
        "age": DIAB_AGE_BUCKETS,
        "readmitted": DIAB_READMIT,
        "change": {"Ch", "No"},
        "diabetesMed": {"Yes", "No"},
        **{c: DIAB_DRUG_LEVELS for c in DIAB_DRUG_COLS},
    },
    numeric_ranges={
        "time_in_hospital": (1, 14),
        "num_lab_procedures": (0, 200),
        "num_procedures": (0, 10),
        "num_medications": (1, 100),
        "number_diagnoses": (1, 20),
        "number_outpatient": (0, 100),
        "number_emergency": (0, 100),
        "number_inpatient": (0, 100),
    },
    consistency_rules={
        # A patient recorded as on diabetes medication should have at least one
        # drug column that is not 'No'.
        "diabetesMed_yes_but_no_drug":
            lambda d: (d["diabetesMed"].str.strip() == "Yes")
                      & (d[DIAB_DRUG_COLS].apply(
                          lambda r: all(str(x).strip() == "No" for x in r), axis=1)),
        # 'change' == 'Ch' asserts a medication change occurred.
        "change_ch_but_no_drug_movement":
            lambda d: (d["change"].str.strip() == "Ch")
                      & (d[DIAB_DRUG_COLS].apply(
                          lambda r: all(str(x).strip() in ("No", "Steady") for x in r),
                          axis=1)),
        # Discharge disposition 11/19/20/21 = expired; such rows cannot be
        # readmitted. A documented leakage trap in this dataset.
        "expired_but_readmitted":
            lambda d: d["discharge_disposition_id"].astype(str).isin(
                          ["11", "19", "20", "21"])
                      & (d["readmitted"].str.strip() != "NO"),
    },
    license_note="CC BY 4.0 (UCI ML Repository, dataset 296)",
    citation=("Strack B, DeShazo JP, Gennings C, Olmo JL, Ventura S, Cios KJ, "
              "Clore JN. Impact of HbA1c measurement on hospital readmission "
              "rates. BioMed Research International. 2014;2014:781670."),
)


# ---------------------------------------------------------------------------
# 3. UCI Online Retail II  -- loaded from Drive; too large to commit to git
#    Chen, Sain & Guo (2012); CC BY 4.0
# ---------------------------------------------------------------------------
def online_retail_spec(path: str) -> DatasetSpec:
    return DatasetSpec(
        name="online_retail_ii",
        path=path,
        read_kwargs={"sep": ","},
        key_columns=(),          # Invoice+StockCode is a composite key
        sentinels=DEFAULT_SENTINELS,
        categorical_domains={},
        numeric_ranges={
            "Quantity": (1, 10_000),      # negatives = returns/cancellations
            "Price": (0.01, 10_000),      # zero/negative = adjustments
        },
        consistency_rules={
            "cancelled_invoice_positive_quantity":
                lambda d: d["Invoice"].astype(str).str.upper().str.startswith("C")
                          & (pd.to_numeric(d["Quantity"], errors="coerce") > 0),
            "negative_quantity_uncancelled":
                lambda d: (~d["Invoice"].astype(str).str.upper().str.startswith("C"))
                          & (pd.to_numeric(d["Quantity"], errors="coerce") < 0),
            "zero_or_negative_price":
                lambda d: pd.to_numeric(d["Price"], errors="coerce") <= 0,
        },
        license_note="CC BY 4.0 (UCI ML Repository, dataset 502)",
        citation=("Chen D, Sain SL, Guo K. Data mining for the online retail "
                  "industry: A case study of RFM model-based customer "
                  "segmentation using data mining. Journal of Database Marketing "
                  "& Customer Strategy Management. 2012;19(3):197-208."),
    )


BANK_SPEC = bank_spec("bank_marketing_small", "/mnt/project/bank.csv")
BANK_FULL_SPEC = bank_spec("bank_marketing_full", "/mnt/project/bankfull.csv")

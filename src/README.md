# src/ — importable modules

Each module is pure Python (plus PySpark/pandas), imported by the notebooks
rather than run directly. Copy the whole directory into your Colab project's
`src/` folder before running any notebook.

| Module | Purpose | Used from |
|---|---|---|
| `ztlf_profiling.py` | Natural-defect census: missingness, uniqueness, domain, range, consistency, representation checks against an unmodified corpus | Notebook 01 |
| `ztlf_specs.py` | Declarative `DatasetSpec` for each corpus: domains, ranges, sentinels, citation, license | All notebooks |
| `ztlf_corruption.py` | Corruption engine with detection-independent ground truth; restricted scoring; the census-informed rule-authoring helper | Notebooks 02–04 |
| `ztlf_plans.py` | Per-corpus corruption plans (which defect classes to inject, at what rate) and the contamination/seed grid | Notebooks 02–04 |
| `ztlf_baselines.py` | Wrappers exposing this framework's gate plus Pandera, Great Expectations, Soda Core, and PyDeequ behind one `gate(df) -> DataFrame[row_id, column]` interface | Notebook 03 |
| `ztlf_downstream.py` | CleanML-style downstream evaluation: train/test split with a clean fixed holdout, quality policies (quarantine/repair), subgroup metrics | Notebook 04 |

## Design contract worth knowing before you read the code

`ztlf_corruption.py` never imports `ztlf_baselines.py`, and no detector
defined in `ztlf_baselines.py` reads the ground-truth mask that
`ztlf_corruption.py` produces. This separation is load-bearing: it is what
makes detection recall capable of falling below 1.0 rather than being 1.0 by
construction. If you extend this code, preserve that boundary.

`natural_defect_keys_full()` in `ztlf_corruption.py` is what the manuscript
calls "restricted scoring" (Section 3.6). It excludes cells that were
defective in the *clean* corpus, before any injection, from the scoring of
both ground truth and detections. Skipping this step reproduces the biased
"naive" numbers in Table 5 rather than the corrected ones.

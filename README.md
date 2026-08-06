# Census-Informed Data Quality Governance for Lakehouse Data Products

Reproducibility artifact for the manuscript *"Census-Informed Data Quality
Governance for Lakehouse Data Products: Rule Authorship, Evaluation Bias,
and Downstream Cost,"* submitted to *Information Systems* (Elsevier).
[![DOI](https://zenodo.org/badge/1324442246.svg)](https://doi.org/10.5281/zenodo.21813665)

## What this is

A portable data-quality pipeline evaluated on three public corpora, testing
whether declarative quality gates actually improve data products — and
finding that the answer depends far more on how the rules were written than
on how faithfully they are enforced.

Three results, in brief:

1. **Scoring detectors against injected-only ground truth understates
   precision by up to 28×** on corpora that already contain defects, worst
   at the low injection rates most benchmarks use.
2. **Given identical rules, three validation engines converge.** The
   reference gate, Pandera, and Soda Core produce equivalent scored
   detections; Great Expectations and PyDeequ differ in capability and
   output semantics. What separates the tools is capability and runtime
   cost, not correctness of the shared predicates.
3. **Rule authorship dominates rule enforcement.** Two rule sets differing
   only in whether their author consulted a data profile before writing
   completeness rules produced training corpora differing by two orders of
   magnitude in retained records, and downstream models differing by up to
   0.151 ROC-AUC — while both configurations reported full compliance.

No quality-gating policy we tested improved downstream model discrimination
relative to applying no gate at all. See the manuscript for the full
argument; this repository lets you reproduce every number in it.

## Reproducing the results

Run the notebooks in `notebooks/` in order, in Google Colab. Each notebook
mounts Google Drive, expects the previous notebook's outputs to already be
present, and writes its own outputs back to Drive.

| Notebook | Produces | Manuscript sections |
|---|---|---|
| `ZTLF_00_environment.ipynb` | Pinned environment, folder structure | 3.9, 4.4 |
| `ZTLF_01_ingestion_and_census.ipynb` | Natural-defect census (T0, T1) | 5.1 |
| `ZTLF_02_corruption_and_detection.ipynb` | Contamination sweep (T3, T4, T5) | 5.2, 5.3 |
| `ZTLF_03_baseline_comparison.ipynb` | Tool comparison (T6, T7, T8) | 5.4 |
| `ZTLF_04_downstream_impact.ipynb` | Rule authorship and downstream evaluation (T8b–T12) | 5.5, 5.6, 5.7, 7 |

Total runtime is roughly two to three hours on a free Colab CPU runtime,
dominated by the Online Retail II corpus (~1M rows) in notebooks 02 and 03.

### Setup

1. Create a folder in your Google Drive, e.g. `MyDrive/reproducibility`.
2. Copy every file in `src/` into `<folder>/src/`.
3. Place the raw data files in `<folder>/data/raw/` (see **Data** below).
4. Open each notebook in Colab and edit the `PROJECT_ROOT` path in its first
   cell to match your folder.
5. Run notebooks 00 through 04 in order.

### Environment

PySpark 3.5.3, delta-spark 3.2.1, OpenJDK 11, pandas, scikit-learn,
matplotlib. Every notebook pins these versions explicitly and writes an
environment manifest (`outputs/logs/environment_manifest.json`) recording
the exact versions, JDK, and a fixed random seed used for that run.

## Data

| Corpus | Rows | License | Included here |
|---|---|---|---|
| [Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing) | 45,211 | CC BY 4.0 | Yes |
| [Diabetes 130-US Hospitals](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) | 101,766 | CC BY 4.0 | Yes |
| [Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) | 1,067,371 | CC BY 4.0 | No |

Due to dataset size and repository storage considerations, Online Retail II is not included in this repository. Please download them directly from the UCI Machine Learning Repository and place them in the data/raw/ directory before running the notebooks.

SHA-256 hashes of the datasets are provided for verification.

## Repository structure

```
src/                      Importable modules (profiling, corruption, baselines, downstream)
notebooks/                Colab notebooks, run in numeric order
data/raw/                 Two redistributable corpora plus checksums.txt
outputs/tables/           Every CSV table referenced in the manuscript (T0-T12)
outputs/figures/          Every figure (Figure1-Figure7), matching manuscript numbering
```

Running the notebooks also creates `outputs/metrics/` (raw, unaggregated
experiment output behind each table) and `outputs/logs/` (environment
manifests) in your own Drive. Those are regenerated on every run and are not
committed here.

Figure files are named `Figure<n>_description.png` and table files
`T<n>_description.csv`, matching the numbering used in the manuscript
directly. Note that the manuscript's figure order is not the order the
figures were generated in — Section 5.2 (scoring bias) precedes Section 5.3
(detection coverage) — so the filenames follow the manuscript, not the
notebooks.

## Key design decisions

**Ground truth is recorded independently of detection logic.** The
corruption engine (`src/ztlf_corruption.py`) never imports or consults the
rule sets used for detection. This is what allows detection performance to
fall below 100% — the earlier version of this study injected and detected
defects with the same logic, making the experiment unfalsifiable by
construction.

**Restricted scoring excludes pre-existing defects.** Scoring a detector
against injected-only ground truth on a corpus that already contains defects
systematically understates precision. `natural_defect_keys_full()` in
`ztlf_corruption.py` computes the exclusion set from the Phase 1 census.

**Rule sets are versioned as data, not code.** The `naive` and
`census_informed` rule sets differ only in which completeness constraints
are admitted (see `outputs/tables/T8b_census_rejected_rules.csv`), so the
same enforcement engine produces directly comparable results under both.

## Citation

If you use this artifact, please cite the manuscript. See `CITATION.cff`.

## License

Code in this repository is released under the Apache License 2.0 (see
`LICENSE`). Bank Marketing and Diabetes 130-US Hospitals data, where
included, retain their original CC BY 4.0 license from the UCI Machine
Learning Repository. Online Retail II is not redistributed.

## Relationship to a prior repository

An earlier repository (`ztlf-lakehouse-framework`) contained a substantially
different study using a single corpus, no baseline comparison, and no
downstream evaluation. That work was withdrawn before peer review. This
repository is an independent, ground-up rebuild and shares no results with
the earlier version. The earlier repository has been archived with a pointer
to this one.

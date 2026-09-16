# Changelog

All notable changes to this reproducibility artifact are recorded here.
Each released version is archived on Zenodo; cite the concept DOI
(https://doi.org/10.5281/zenodo.21813665) rather than a version-specific
one, so citations resolve to the most recent release.

## [1.0.2] — 2026-08

### Fixed
- Restored `data/raw/checksums.txt`, which was omitted from the initial
  upload. The manuscript's data availability statement refers to it, so its
  absence left a broken reproducibility claim. It now lists verified SHA-256
  digests for all four source files, including Online Retail II.
- Corrected the README's description of the figure file naming scheme to
  match the files actually committed.

### Changed
- Figure files renamed to match the manuscript's figure numbering. The
  manuscript presents the scoring-protocol result (Section 5.2) before the
  detection-coverage result (Section 5.3), so the generation order and the
  presentation order differ. Files now follow the manuscript.

## [1.0.1] — 2026-08

### Fixed
- **Dataset licensing corrected.** Online Retail II was previously described
  as non-commercial-use-only. It is licensed CC BY 4.0, the same as the other
  two corpora. The error appeared in the README, `data/raw/checksums.txt`,
  `src/ztlf_specs.py`, and the generated
  `outputs/tables/T0_dataset_provenance.csv`. All four were corrected, and
  the provenance table was regenerated so the license field is accurate at
  source rather than only in documentation. The dataset remains excluded
  from the repository, but for file size (109 MB) rather than licensing.
- Corrected the README's summary of the tool-comparison result. It claimed
  four tools converged on identical detections; three converged (the
  reference gate, Pandera, and Soda Core) while Great Expectations and
  PyDeequ differed in capability and output semantics. The manuscript had
  already been corrected; the README had not.

## [1.0.0] — 2026-08

Initial public release accompanying the manuscript submission.

### Included
- Five Colab notebooks covering environment setup, the natural-defect
  census, controlled corruption and detection, the baseline tool
  comparison, and the downstream model evaluation.
- Six Python modules implementing the census, corruption engine with
  detection-independent ground truth, per-corpus corruption plans, the
  unified detector interface wrapping four open-source validation tools,
  and the downstream evaluation harness.
- Fifteen result tables (T0–T12) and seven figures, all generated from
  measured results by the analysis code.
- Two of the three source corpora, with SHA-256 digests for all four.

### Notes
- Notebooks are committed with outputs cleared and Colab session metadata
  stripped.
- Execution is single-node. The artifact makes no throughput, latency, or
  scalability claims.

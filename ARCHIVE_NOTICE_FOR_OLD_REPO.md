# ⚠️ Archived — superseded by a ground-up rebuild

This repository accompanied an earlier manuscript submission that was
**withdrawn before peer review**. That study had significant methodological
weaknesses, identified through subsequent review:

- The evaluation injected synthetic defects and then detected them with
  rules derived from the same logic, making precision and recall equal to
  1.0 by construction rather than by measurement.
- No baseline comparison against existing data-quality tools was performed.
- Only one dataset (10,000 records) was evaluated.
- The "Zero-Trust" framing described a configuration of managed-identity
  storage access but included no implemented, adversarially tested access
  control layer.
- No downstream model evaluation was performed, so the "AI-ready" claim in
  the title was asserted rather than tested.

**This work has been superseded by a complete, independent rebuild:**

➡️ **https://github.com/kallamrameshbabu/REPO_NAME_PLACEHOLDER**

The new study:
- Evaluates three public corpora (45,211 / 101,766 / ~1,067,371 records)
  spanning marketing, clinical, and retail domains
- Uses a corruption engine whose ground truth is recorded independently of
  detection logic, so precision and recall can fall below 1.0
- Compares against four established open-source validation tools under
  identical rule sets
- Evaluates downstream model impact under a clean-holdout protocol
- Finds that rule authorship — not enforcement — is the dominant source of
  variation in pipeline outcomes, and reports this as the paper's central
  contribution

This repository is retained for historical transparency and is not
maintained. Please cite and build upon the new repository instead.

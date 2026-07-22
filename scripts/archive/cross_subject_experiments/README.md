# Archived Cross-Subject Experiments

This directory contains completed cross-subject experiment variants retained for reproducibility.

Active runner:

    python -m scripts.run_cross_subject_evaluation

Active configuration:

- Training subjects: A01T-A08T
- Unseen test subject: A09T
- Subject-wise Euclidean Alignment
- Frequency bands: 8-30 Hz
- Ten CSP components per band
- PCA retaining 90% variance
- Shrinkage LDA

Recorded result:

- Accuracy: 66.0%
- Cohen kappa: 0.546

Archived files contain CSP component, PCA, no-PCA, and subject z-score comparisons.

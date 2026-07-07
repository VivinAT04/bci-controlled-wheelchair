# Subject Variability Analysis

## Overview

The classification results show considerable inter-subject variability across the nine participants.

## High-performing subjects

Subjects A03, A07 and A08 consistently achieved accuracies above 80% across most models.

These participants likely produced more stable and discriminative motor imagery EEG patterns, allowing CSP and FBCSP to learn more effective spatial filters.

## Low-performing subjects

Subjects A04 and A06 remained close to chance level across nearly all models.

Possible reasons include:

- weaker motor imagery responses
- higher EEG noise or physiological artifacts
- larger intra-subject variability
- lower class separability
- limitations of CSP assumptions for these subjects

## Effect of FBCSP

Subject A05 showed the largest improvement.

- CSP: **38.2%**
- FBCSP: **72.6%**

This indicates that discriminative information existed in specific frequency bands which standard CSP could not effectively capture.

## Overall Observation

The results demonstrate strong inter-subject variability, which is common in EEG-based motor imagery classification. Advanced feature extraction methods such as FBCSP substantially improved performance for some participants but had limited impact on consistently difficult subjects. This motivates investigating adaptive approaches such as Riemannian geometry-based classifiers and learnable CSP methods in future work.

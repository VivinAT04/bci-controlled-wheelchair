"""Run McNemar tests using saved within-subject predictions."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2


PREDICTIONS_PATH = Path(
    "results/within_subject/statistical_tests/mcnemar_predictions.csv"
)

RESULTS_PATH = Path(
    "results/within_subject/statistical_tests/mcnemar_results.csv"
)

ALPHA = 0.05

COMPARISONS = [
    ("csp_pred", "tuned_csp_pred", "CSP", "Tuned CSP"),
    ("csp_pred", "fbcsp_pred", "CSP", "FBCSP"),
    ("tuned_csp_pred", "fbcsp_pred", "Tuned CSP", "FBCSP"),
]

BONFERRONI_ALPHA = ALPHA / len(COMPARISONS)


def mcnemar_test(
    y_true,
    pred_a,
    pred_b,
    name_a,
    name_b,
):
    """Compare two classifiers using McNemar's test."""

    a_correct = pred_a == y_true
    b_correct = pred_b == y_true

    both_correct = np.sum(a_correct & b_correct)
    a_correct_b_wrong = np.sum(a_correct & ~b_correct)
    a_wrong_b_correct = np.sum(~a_correct & b_correct)
    both_wrong = np.sum(~a_correct & ~b_correct)

    b = a_correct_b_wrong
    c = a_wrong_b_correct

    if b + c == 0:
        statistic = 0.0
        p_value = 1.0
    else:
        statistic = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = chi2.sf(statistic, df=1)

    return {
        "Comparison": f"{name_a} vs {name_b}",
        "Both Correct": both_correct,
        f"{name_a} Correct, {name_b} Wrong": a_correct_b_wrong,
        f"{name_a} Wrong, {name_b} Correct": a_wrong_b_correct,
        "Both Wrong": both_wrong,
        "Chi-square": statistic,
        "p-value": p_value,
        "Significant p<0.05": p_value < ALPHA,
        (
            "Significant "
            f"Bonferroni p<{BONFERRONI_ALPHA:.4f}"
        ): p_value < BONFERRONI_ALPHA,
    }


def main():
    """Load predictions, run McNemar tests, and save results."""

    print("\n========================================")
    print("McNemar Statistical Tests")
    print("========================================")

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {PREDICTIONS_PATH}"
        )

    predictions = pd.read_csv(PREDICTIONS_PATH)

    required_columns = {
        "y_true",
        "csp_pred",
        "tuned_csp_pred",
        "fbcsp_pred",
    }

    missing = required_columns - set(predictions.columns)

    if missing:
        raise ValueError(
            f"Missing prediction columns: {sorted(missing)}"
        )

    y_true = predictions["y_true"].to_numpy()

    results = []

    for column_a, column_b, name_a, name_b in COMPARISONS:
        result = mcnemar_test(
            y_true,
            predictions[column_a].to_numpy(),
            predictions[column_b].to_numpy(),
            name_a,
            name_b,
        )

        results.append(result)

    results_dataframe = pd.DataFrame(results)

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_dataframe.to_csv(
        RESULTS_PATH,
        index=False,
    )

    print(results_dataframe.to_string(index=False))

    print("\nSaved:")
    print(RESULTS_PATH)


if __name__ == "__main__":
    main()

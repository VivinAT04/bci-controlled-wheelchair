"""
Run Wilcoxon signed-rank tests between within-subject classifiers.

Run from the project root:

    python -m scripts.statistical_analysis.run_wilcoxon_tests
"""

from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon


OUTPUT_PATH = Path("results/within_subject/statistical_tests/wilcoxon_results.csv")

MODELS = {
    "CSP": [
        78.5,
        51.4,
        85.8,
        49.0,
        38.2,
        49.0,
        74.7,
        80.6,
        72.9,
    ],
    "FBCSP": [
        77.4,
        53.8,
        85.1,
        49.7,
        72.6,
        47.6,
        87.2,
        82.3,
        76.7,
    ],
    "CSP+SVM": [
        80.2,
        52.4,
        84.0,
        47.6,
        41.7,
        49.7,
        75.7,
        81.6,
        73.6,
    ],
    "FBCSP+SVM": [
        78.1,
        55.9,
        85.4,
        52.1,
        69.1,
        49.7,
        83.0,
        83.7,
        79.2,
    ],
}

COMPARISONS = [
    ("CSP", "FBCSP"),
    ("CSP", "CSP+SVM"),
    ("FBCSP", "FBCSP+SVM"),
    ("CSP", "FBCSP+SVM"),
]

ALPHA = 0.05
BONFERRONI_ALPHA = ALPHA / len(COMPARISONS)


def run_wilcoxon_tests() -> pd.DataFrame:
    """Run all configured Wilcoxon signed-rank comparisons."""

    results = []

    for model_1, model_2 in COMPARISONS:
        statistic, p_value = wilcoxon(
            MODELS[model_1],
            MODELS[model_2],
        )

        results.append(
            {
                "Comparison": f"{model_1} vs {model_2}",
                "Statistic": statistic,
                "p-value": p_value,
                "Significant (p<0.05)": p_value < ALPHA,
                (
                    "Significant "
                    f"(Bonferroni p<{BONFERRONI_ALPHA:.4f})"
                ): p_value < BONFERRONI_ALPHA,
            }
        )

    return pd.DataFrame(results)


def main() -> None:
    """Run the tests, print the results, and save them to CSV."""

    print("\n========================================")
    print("Wilcoxon Signed-Rank Test Results")
    print("========================================")

    results_dataframe = run_wilcoxon_tests()

    print(results_dataframe.to_string(index=False))

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_dataframe.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\nSaved:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()

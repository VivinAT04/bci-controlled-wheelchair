"""
Final statistical comparison of cross-session classifiers.

Protocol
--------
Train:
    pooled A01T-A09T

Test:
    A01E-A09E

Analyses
--------
1. Trial-level exact McNemar tests using paired predictions.
2. Subject-level Wilcoxon signed-rank tests using the nine
   evaluation-subject accuracies.
3. Holm-Bonferroni correction within each family of pairwise tests.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon


RESULT_ROOT = Path("results/cross_session")

MODEL_FILES = {
    "CSP + LDA": {
        "predictions": (
            RESULT_ROOT
            / "csp_lda"
            / "csp_lda_cross_session_predictions.csv"
        ),
        "subjects": (
            RESULT_ROOT
            / "csp_lda"
            / "csp_lda_cross_session_subject_results.csv"
        ),
    },
    "Tuned CSP + LDA": {
        "predictions": (
            RESULT_ROOT
            / "tuned_csp_lda"
            / "tuned_csp_lda_cross_session_predictions.csv"
        ),
        "subjects": (
            RESULT_ROOT
            / "tuned_csp_lda"
            / "tuned_csp_lda_cross_session_subject_results.csv"
        ),
    },
    "FBCSP + LDA": {
        "predictions": (
            RESULT_ROOT
            / "fbcsp_lda"
            / "fbcsp_lda_cross_session_predictions.csv"
        ),
        "subjects": (
            RESULT_ROOT
            / "fbcsp_lda"
            / "fbcsp_lda_cross_session_subject_results.csv"
        ),
    },
    "CSP + RBF-SVM": {
        "predictions": (
            RESULT_ROOT
            / "csp_rbf_svm"
            / "csp_rbf_svm_cross_session_predictions.csv"
        ),
        "subjects": (
            RESULT_ROOT
            / "csp_rbf_svm"
            / "csp_rbf_svm_cross_session_subject_results.csv"
        ),
    },
    "FBCSP + RBF-SVM": {
        "predictions": (
            RESULT_ROOT
            / "fbcsp_rbf_svm"
            / "fbcsp_rbf_svm_cross_session_predictions.csv"
        ),
        "subjects": (
            RESULT_ROOT
            / "fbcsp_rbf_svm"
            / "fbcsp_rbf_svm_cross_session_subject_results.csv"
        ),
    },
    "Riemannian MDM": {
        "predictions": (
            RESULT_ROOT
            / "riemannian"
            / "mdm"
            / "riemannian_mdm_cross_session_predictions.csv"
        ),
        "subjects": (
            RESULT_ROOT
            / "riemannian"
            / "mdm"
            / "riemannian_mdm_cross_session_subject_results.csv"
        ),
    },
    "Riemannian TS + LDA": {
        "predictions": (
            RESULT_ROOT
            / "riemannian"
            / "tangent_lda"
            / "tangent_lda_cross_session_predictions.csv"
        ),
        "subjects": (
            RESULT_ROOT
            / "riemannian"
            / "tangent_lda"
            / "tangent_lda_cross_session_subject_results.csv"
        ),
    },
    "Filter-Bank Riemannian": {
        "predictions": (
            RESULT_ROOT
            / "riemannian"
            / "filterbank"
            / "filterbank_riemannian_cross_session_predictions.csv"
        ),
        "subjects": (
            RESULT_ROOT
            / "riemannian"
            / "filterbank"
            / "filterbank_riemannian_cross_session_subject_results.csv"
        ),
    },
    "Autoencoder + RBF-SVM": {
        "predictions": (
            RESULT_ROOT
            / "autoencoder_rbf_svm"
            / "autoencoder_rbf_svm_cross_session_predictions.csv"
        ),
        "subjects": (
            RESULT_ROOT
            / "autoencoder_rbf_svm"
            / "autoencoder_rbf_svm_cross_session_subject_results.csv"
        ),
    },
    "Supervised Autoencoder + RBF-SVM": {
        "predictions": (
            RESULT_ROOT
            / "supervised_autoencoder_rbf_svm"
            / "supervised_autoencoder_rbf_svm_cross_session_predictions.csv"
        ),
        "subjects": (
            RESULT_ROOT
            / "supervised_autoencoder_rbf_svm"
            / "supervised_autoencoder_rbf_svm_cross_session_subject_results.csv"
        ),
    },
    "EEGNet": {
        "predictions": (
            RESULT_ROOT
            / "eegnet"
            / "eegnet_cross_session_predictions.csv"
        ),
        "subjects": (
            RESULT_ROOT
            / "eegnet"
            / "eegnet_cross_session_subject_results.csv"
        ),
    },
    "EA + FBCSP + Shrinkage LDA": {
        "predictions": (
            RESULT_ROOT
            / "ea_fbcsp_lda"
            / "ea_fbcsp_lda_cross_session_predictions.csv"
        ),
        "subjects": (
            RESULT_ROOT
            / "ea_fbcsp_lda"
            / "ea_fbcsp_lda_cross_session_subject_results.csv"
        ),
    },
}

OUTPUT_DIR = Path(
    "results/statistical_analysis/cross_session"
)

MCNEMAR_OUTPUT = (
    OUTPUT_DIR
    / "cross_session_mcnemar_results.csv"
)

WILCOXON_OUTPUT = (
    OUTPUT_DIR
    / "cross_session_wilcoxon_results.csv"
)


def find_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> str:
    for candidate in candidates:
        if candidate in dataframe.columns:
            return candidate

    raise KeyError(
        "Could not identify required column. "
        f"Available columns: {list(dataframe.columns)}"
    )


def holm_adjust(
    p_values: np.ndarray,
) -> np.ndarray:
    order = np.argsort(p_values)
    m = len(p_values)

    adjusted = np.empty(
        m,
        dtype=float,
    )

    running_max = 0.0

    for rank, index in enumerate(
        order,
        start=1,
    ):
        value = min(
            1.0,
            (m - rank + 1)
            * float(p_values[index]),
        )

        running_max = max(
            running_max,
            value,
        )

        adjusted[index] = running_max

    return adjusted


def load_correctness(
    path: Path,
) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {path}"
        )

    dataframe = pd.read_csv(path)

    true_column = find_column(
        dataframe,
        [
            "true_label",
            "true_class",
            "y_true",
            "actual",
            "label",
        ],
    )

    predicted_column = find_column(
        dataframe,
        [
            "predicted_label",
            "predicted_class",
            "y_pred",
            "prediction",
        ],
    )

    return (
        dataframe[true_column]
        .astype(str)
        .to_numpy()
        ==
        dataframe[predicted_column]
        .astype(str)
        .to_numpy()
    )


def load_subject_accuracies(
    path: Path,
) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(
            f"Subject results not found: {path}"
        )

    dataframe = pd.read_csv(path)

    subject_column = find_column(
        dataframe,
        [
            "subject",
            "test_subject",
            "held_out_subject",
            "subject_name",
        ],
    )

    accuracy_column = find_column(
        dataframe,
        [
            "accuracy",
            "test_accuracy",
            "accuracy_percent",
            "accuracy_percentage",
        ],
    )

    series = dataframe.set_index(
        subject_column
    )[accuracy_column].astype(float)

    if series.max() > 1.0:
        series = series / 100.0

    return series.sort_index()


def compare_mcnemar(
    name_a: str,
    correct_a: np.ndarray,
    name_b: str,
    correct_b: np.ndarray,
) -> dict:
    if len(correct_a) != len(correct_b):
        raise ValueError(
            f"Prediction lengths differ: "
            f"{name_a}={len(correct_a)}, "
            f"{name_b}={len(correct_b)}"
        )

    both_correct = int(
        np.sum(correct_a & correct_b)
    )

    a_correct_b_wrong = int(
        np.sum(correct_a & ~correct_b)
    )

    a_wrong_b_correct = int(
        np.sum(~correct_a & correct_b)
    )

    both_wrong = int(
        np.sum(~correct_a & ~correct_b)
    )

    discordant = (
        a_correct_b_wrong
        + a_wrong_b_correct
    )

    if discordant == 0:
        exact_p = 1.0
    else:
        exact_p = float(
            binomtest(
                a_correct_b_wrong,
                n=discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )

    accuracy_a = float(
        np.mean(correct_a)
    )

    accuracy_b = float(
        np.mean(correct_b)
    )

    return {
        "model_a": name_a,
        "model_b": name_b,
        "n_trials": len(correct_a),
        "model_a_accuracy_percent": (
            accuracy_a * 100.0
        ),
        "model_b_accuracy_percent": (
            accuracy_b * 100.0
        ),
        "difference_pp_a_minus_b": (
            accuracy_a - accuracy_b
        ) * 100.0,
        "both_correct": both_correct,
        "a_correct_b_wrong": (
            a_correct_b_wrong
        ),
        "a_wrong_b_correct": (
            a_wrong_b_correct
        ),
        "both_wrong": both_wrong,
        "discordant_pairs": discordant,
        "mcnemar_p_exact": exact_p,
    }


def compare_wilcoxon(
    name_a: str,
    values_a: pd.Series,
    name_b: str,
    values_b: pd.Series,
) -> dict:
    common_subjects = (
        values_a.index.intersection(
            values_b.index
        )
    )

    a = values_a.loc[
        common_subjects
    ].to_numpy()

    b = values_b.loc[
        common_subjects
    ].to_numpy()

    differences = a - b

    if np.allclose(
        differences,
        0.0,
    ):
        statistic = 0.0
        p_value = 1.0
    else:
        result = wilcoxon(
            a,
            b,
            alternative="two-sided",
            zero_method="wilcox",
            correction=False,
            method="auto",
        )

        statistic = float(
            result.statistic
        )

        p_value = float(
            result.pvalue
        )

    return {
        "model_a": name_a,
        "model_b": name_b,
        "n_subjects": len(
            common_subjects
        ),
        "model_a_mean_accuracy_percent": (
            np.mean(a) * 100.0
        ),
        "model_b_mean_accuracy_percent": (
            np.mean(b) * 100.0
        ),
        "mean_difference_pp_a_minus_b": (
            np.mean(differences)
            * 100.0
        ),
        "median_difference_pp_a_minus_b": (
            np.median(differences)
            * 100.0
        ),
        "wilcoxon_statistic": statistic,
        "wilcoxon_p": p_value,
    }


def run_mcnemar_analysis() -> pd.DataFrame:
    correctness = {}

    print()
    print(
        "Trial-level McNemar inputs"
    )
    print(
        "-" * 80
    )

    for model_name, paths in MODEL_FILES.items():
        values = load_correctness(
            paths["predictions"]
        )

        correctness[
            model_name
        ] = values

        print(
            f"{model_name:<38} "
            f"{np.mean(values) * 100:6.2f}% "
            f"n={len(values)}"
        )

    lengths = {
        len(values)
        for values in correctness.values()
    }

    if len(lengths) != 1:
        raise RuntimeError(
            "Prediction files do not contain "
            "the same number of paired trials."
        )

    rows = []

    for model_a, model_b in combinations(
        MODEL_FILES.keys(),
        2,
    ):
        rows.append(
            compare_mcnemar(
                model_a,
                correctness[model_a],
                model_b,
                correctness[model_b],
            )
        )

    results = pd.DataFrame(rows)

    results[
        "holm_adjusted_p"
    ] = holm_adjust(
        results[
            "mcnemar_p_exact"
        ].to_numpy()
    )

    results[
        "significant_0_05_raw"
    ] = (
        results[
            "mcnemar_p_exact"
        ]
        < 0.05
    )

    results[
        "significant_0_05_holm"
    ] = (
        results[
            "holm_adjusted_p"
        ]
        < 0.05
    )

    return results


def run_wilcoxon_analysis() -> pd.DataFrame:
    subject_results = {}

    print()
    print(
        "Subject-level Wilcoxon inputs"
    )
    print(
        "-" * 80
    )

    for model_name, paths in MODEL_FILES.items():
        values = load_subject_accuracies(
            paths["subjects"]
        )

        subject_results[
            model_name
        ] = values

        print(
            f"{model_name:<38} "
            f"{values.mean() * 100:6.2f}% "
            f"subjects={len(values)}"
        )

    rows = []

    for model_a, model_b in combinations(
        MODEL_FILES.keys(),
        2,
    ):
        rows.append(
            compare_wilcoxon(
                model_a,
                subject_results[model_a],
                model_b,
                subject_results[model_b],
            )
        )

    results = pd.DataFrame(rows)

    results[
        "holm_adjusted_p"
    ] = holm_adjust(
        results[
            "wilcoxon_p"
        ].to_numpy()
    )

    results[
        "significant_0_05_raw"
    ] = (
        results[
            "wilcoxon_p"
        ]
        < 0.05
    )

    results[
        "significant_0_05_holm"
    ] = (
        results[
            "holm_adjusted_p"
        ]
        < 0.05
    )

    return results


def print_significant(
    results: pd.DataFrame,
    p_column: str,
    title: str,
) -> None:
    print()
    print(title)
    print(
        "-" * 80
    )

    significant = results[
        results[
            "significant_0_05_holm"
        ]
    ]

    if significant.empty:
        print("None.")
        return

    columns = [
        "model_a",
        "model_b",
        p_column,
        "holm_adjusted_p",
    ]

    print(
        significant[
            columns
        ].to_string(
            index=False
        )
    )


def main() -> None:
    print(
        "=" * 80
    )
    print(
        "FINAL CROSS-SESSION STATISTICAL ANALYSIS"
    )
    print(
        "=" * 80
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    mcnemar_results = (
        run_mcnemar_analysis()
    )

    mcnemar_results.to_csv(
        MCNEMAR_OUTPUT,
        index=False,
    )

    wilcoxon_results = (
        run_wilcoxon_analysis()
    )

    wilcoxon_results.to_csv(
        WILCOXON_OUTPUT,
        index=False,
    )

    print()
    print(
        "=" * 80
    )
    print(
        "OUTPUT FILES"
    )
    print(
        "=" * 80
    )

    print(
        MCNEMAR_OUTPUT
    )

    print(
        WILCOXON_OUTPUT
    )

    print_significant(
        mcnemar_results,
        "mcnemar_p_exact",
        (
            "McNemar comparisons significant "
            "after Holm correction"
        ),
    )

    print_significant(
        wilcoxon_results,
        "wilcoxon_p",
        (
            "Subject-level Wilcoxon comparisons "
            "significant after Holm correction"
        ),
    )


if __name__ == "__main__":
    main()

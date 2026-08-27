"""
Final statistical comparison of cross-subject classifiers.

Protocol
--------
Strict T-to-E LOSO:

For each held-out subject:
    Train on the other eight subjects' T sessions.
    Test on the held-out subject's E session.

Analysis
--------
Subject-level paired Wilcoxon signed-rank tests using the
nine held-out-subject accuracies.

Holm-Bonferroni correction is applied across all pairwise tests.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


RESULT_ROOT = Path(
    "results/cross_subject"
)

MODEL_FILES = {
    "CSP + LDA": (
        RESULT_ROOT
        / "csp_lda"
        / "csp_lda_cross_subject_subject_results.csv"
    ),
    "Tuned CSP + LDA": (
        RESULT_ROOT
        / "tuned_csp_lda"
        / "tuned_csp_lda_cross_subject_subject_results.csv"
    ),
    "FBCSP + LDA": (
        RESULT_ROOT
        / "fbcsp_lda"
        / "fbcsp_lda_cross_subject_subject_results.csv"
    ),
    "CSP + RBF-SVM": (
        RESULT_ROOT
        / "csp_rbf_svm"
        / "csp_rbf_svm_cross_subject_subject_results.csv"
    ),
    "FBCSP + RBF-SVM": (
        RESULT_ROOT
        / "fbcsp_rbf_svm"
        / "fbcsp_rbf_svm_cross_subject_subject_results.csv"
    ),
    "Riemannian MDM": (
        RESULT_ROOT
        / "riemannian_mdm"
        / "riemannian_mdm_cross_subject_subject_results.csv"
    ),
    "Riemannian TS + LDA": (
        RESULT_ROOT
        / "riemannian_tangent_lda"
        / "riemannian_tangent_lda_cross_subject_subject_results.csv"
    ),
    "Filter-Bank Riemannian": (
        RESULT_ROOT
        / "filterbank_riemannian"
        / "filterbank_riemannian_cross_subject_subject_results.csv"
    ),
    "Autoencoder + RBF-SVM": (
        RESULT_ROOT
        / "autoencoder_rbf_svm"
        / "autoencoder_rbf_svm_cross_subject_subject_results.csv"
    ),
    "Supervised Autoencoder + RBF-SVM": (
        RESULT_ROOT
        / "supervised_autoencoder_rbf_svm"
        / "supervised_autoencoder_rbf_svm_cross_subject_subject_results.csv"
    ),

    # Final improved strict LOSO EEGNet result.
    "EEGNet": (
        RESULT_ROOT
        / "eegnet"
        / "eegnet_loso_improved"
        / "subject_summary.csv"
    ),

    "EA + FBCSP + Shrinkage LDA": (
        RESULT_ROOT
        / "ea_fbcsp_lda"
        / "ea_fbcsp_lda_cross_subject_subject_results.csv"
    ),
}


OUTPUT_DIR = Path(
    "results/statistical_analysis/cross_subject"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "cross_subject_wilcoxon_results.csv"
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
            "evaluation_subject",
        ],
    )

    accuracy_column = find_column(
        dataframe,
        [
            "accuracy",
            "test_accuracy",
            "accuracy_percent",
            "accuracy_percentage",
            "mean_accuracy",
            "mean_accuracy_percent",
        ],
    )

    series = dataframe.set_index(
        subject_column
    )[accuracy_column].astype(float)

    if series.max() > 1.0:
        series = series / 100.0

    return series.sort_index()


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
        adjusted_value = min(
            1.0,
            (m - rank + 1)
            * float(
                p_values[index]
            ),
        )

        running_max = max(
            running_max,
            adjusted_value,
        )

        adjusted[
            index
        ] = running_max

    return adjusted


def compare_models(
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

    if len(common_subjects) != 9:
        raise ValueError(
            f"{name_a} vs {name_b}: "
            f"expected 9 matched subjects, "
            f"found {len(common_subjects)}."
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


def main() -> None:
    print(
        "=" * 80
    )
    print(
        "FINAL CROSS-SUBJECT STATISTICAL ANALYSIS"
    )
    print(
        "=" * 80
    )

    subject_results = {}

    for model_name, path in MODEL_FILES.items():
        values = load_subject_accuracies(
            path
        )

        if len(values) != 9:
            raise ValueError(
                f"{model_name}: expected 9 "
                f"subject results, found "
                f"{len(values)}."
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
            compare_models(
                model_a,
                subject_results[model_a],
                model_b,
                subject_results[model_b],
            )
        )

    results = pd.DataFrame(
        rows
    )

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

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        "Saved:"
    )
    print(
        OUTPUT_PATH
    )

    print()
    print(
        "Significant comparisons "
        "after Holm correction:"
    )

    significant = results[
        results[
            "significant_0_05_holm"
        ]
    ]

    if significant.empty:
        print(
            "None."
        )
    else:
        print(
            significant[
                [
                    "model_a",
                    "model_b",
                    "mean_difference_pp_a_minus_b",
                    "wilcoxon_p",
                    "holm_adjusted_p",
                ]
            ].to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()

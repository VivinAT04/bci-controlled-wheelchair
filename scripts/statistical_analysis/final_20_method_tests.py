"""
Final dissertation statistical analysis.

20 final classification methods.

Cross-Session:
    - Exact paired McNemar tests on 2,592 evaluation trials.
    - Wilcoxon signed-rank tests on 9 subject accuracies.

Cross-Subject:
    - Wilcoxon signed-rank tests on 9 held-out-subject accuracies.

Multiple-comparison correction:
    - Holm step-down correction.

Protocol-specific winners:
    Cross-Session:
        EA + FBCSP + RBF-SVM
    Cross-Subject:
        EA + CSP + Shrinkage LDA

The script validates every selected statistical input against:
results/dissertation_figure_data/canonical_20_method_results.csv
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import binomtest, wilcoxon


# ======================================================================
# GLOBAL CONFIG
# ======================================================================

CANONICAL = Path(
    "results/dissertation_figure_data/"
    "canonical_20_method_results.csv"
)

CS_OUTPUT = Path(
    "results/statistical_analysis/cross_session"
)

XS_OUTPUT = Path(
    "results/statistical_analysis/cross_subject"
)

FIGURE_DATA = Path(
    "results/dissertation_figure_data"
)


METHOD_ORDER = [
    "CSP + LDA",
    "CSP + RBF-SVM",
    "Tuned CSP + LDA",
    "Tuned CSP + RBF-SVM",
    "FBCSP + LDA",
    "FBCSP + RBF-SVM",
    "Riemannian MDM",
    "Riemannian TS + Shrinkage LDA",
    "Riemannian TS + RBF-SVM",
    "Filter-Bank Riemannian + Shrinkage LDA",
    "Filter-Bank Riemannian + RBF-SVM",
    "Autoencoder + LDA",
    "Autoencoder + RBF-SVM",
    "Supervised Autoencoder + LDA",
    "Supervised Autoencoder + RBF-SVM",
    "EEGNet",
    "EA + CSP + Shrinkage LDA",
    "EA + CSP + RBF-SVM",
    "EA + FBCSP + Shrinkage LDA",
    "EA + FBCSP + RBF-SVM",
]


BEST_CROSS_SESSION = (
    "EA + FBCSP + RBF-SVM"
)

BEST_CROSS_SUBJECT = (
    "EA + CSP + Shrinkage LDA"
)


# ======================================================================
# EXPLICIT FINAL CROSS-SESSION SOURCES
# ======================================================================

CS_PREDICTIONS = {

    "CSP + LDA":
        Path(
            "results/cross_session/csp_lda/"
            "csp_lda_cross_session_predictions.csv"
        ),

    "CSP + RBF-SVM":
        Path(
            "results/cross_session/csp_rbf_svm/"
            "csp_rbf_svm_cross_session_predictions.csv"
        ),

    "Tuned CSP + LDA":
        Path(
            "results/cross_session/tuned_csp_lda/"
            "tuned_csp_lda_cross_session_predictions.csv"
        ),

    "Tuned CSP + RBF-SVM":
        Path(
            "results/cross_session/tuned_csp_rbf_svm/"
            "tuned_csp_rbf_svm_cross_session_predictions.csv"
        ),

    "FBCSP + LDA":
        Path(
            "results/cross_session/fbcsp_lda/"
            "fbcsp_lda_cross_session_predictions.csv"
        ),

    "FBCSP + RBF-SVM":
        Path(
            "results/cross_session/fbcsp_rbf_svm/"
            "fbcsp_rbf_svm_cross_session_predictions.csv"
        ),

    "Riemannian MDM":
        Path(
            "results/cross_session/riemannian/mdm/"
            "riemannian_mdm_cross_session_predictions.csv"
        ),

    "Riemannian TS + Shrinkage LDA":
        Path(
            "results/cross_session/riemannian/tangent_lda/"
            "tangent_lda_cross_session_predictions.csv"
        ),

    "Riemannian TS + RBF-SVM":
        Path(
            "results/cross_session/riemannian/tangent_rbf_svm/"
            "tangent_rbf_svm_cross_session_predictions.csv"
        ),

    "Filter-Bank Riemannian + Shrinkage LDA":
        Path(
            "results/cross_session/riemannian/filterbank/lda/"
            "filterbank_riemannian_lda_cross_session_predictions.csv"
        ),

    "Filter-Bank Riemannian + RBF-SVM":
        Path(
            "results/cross_session/riemannian/filterbank/svm/"
            "filterbank_riemannian_svm_cross_session_predictions.csv"
        ),

    "Autoencoder + LDA":
        Path(
            "results/cross_session/autoencoder_lda/"
            "autoencoder_lda_cross_session_predictions.csv"
        ),

    "Autoencoder + RBF-SVM":
        Path(
            "results/cross_session/autoencoder_rbf_svm/"
            "autoencoder_rbf_svm_cross_session_predictions.csv"
        ),

    "Supervised Autoencoder + LDA":
        Path(
            "results/cross_session/supervised_autoencoder_lda/"
            "supervised_autoencoder_lda_cross_session_predictions.csv"
        ),

    "Supervised Autoencoder + RBF-SVM":
        Path(
            "results/cross_session/supervised_autoencoder_rbf_svm/"
            "supervised_autoencoder_rbf_svm_cross_session_predictions.csv"
        ),

    "EEGNet":
        Path(
            "results/cross_session/eegnet/"
            "eegnet_cross_session_predictions.csv"
        ),

    "EA + CSP + Shrinkage LDA":
        Path(
            "results/cross_session/euclidean_alignment/csp/lda/"
            "ea_csp_lda_cross_session_predictions.csv"
        ),

    "EA + CSP + RBF-SVM":
        Path(
            "results/cross_session/euclidean_alignment/csp/svm/"
            "ea_csp_svm_cross_session_predictions.csv"
        ),

    "EA + FBCSP + Shrinkage LDA":
        Path(
            "results/cross_session/euclidean_alignment/fbcsp/lda/"
            "ea_fbcsp_lda_cross_session_predictions.csv"
        ),

    "EA + FBCSP + RBF-SVM":
        Path(
            "results/cross_session/euclidean_alignment/fbcsp/svm/"
            "ea_fbcsp_svm_cross_session_predictions.csv"
        ),
}


CS_SUBJECT_RESULTS = {

    "CSP + LDA":
        Path(
            "results/cross_session/csp_lda/"
            "csp_lda_cross_session_subject_results.csv"
        ),

    "CSP + RBF-SVM":
        Path(
            "results/cross_session/csp_rbf_svm/"
            "csp_rbf_svm_cross_session_subject_results.csv"
        ),

    "Tuned CSP + LDA":
        Path(
            "results/cross_session/tuned_csp_lda/"
            "tuned_csp_lda_cross_session_subject_results.csv"
        ),

    "Tuned CSP + RBF-SVM":
        Path(
            "results/cross_session/tuned_csp_rbf_svm/"
            "tuned_csp_rbf_svm_cross_session_subject_results.csv"
        ),

    "FBCSP + LDA":
        Path(
            "results/cross_session/fbcsp_lda/"
            "fbcsp_lda_cross_session_subject_results.csv"
        ),

    "FBCSP + RBF-SVM":
        Path(
            "results/cross_session/fbcsp_rbf_svm/"
            "fbcsp_rbf_svm_cross_session_subject_results.csv"
        ),

    "Riemannian MDM":
        Path(
            "results/cross_session/riemannian/mdm/"
            "riemannian_mdm_cross_session_subject_results.csv"
        ),

    "Riemannian TS + Shrinkage LDA":
        Path(
            "results/cross_session/riemannian/tangent_lda/"
            "tangent_lda_cross_session_subject_results.csv"
        ),

    "Riemannian TS + RBF-SVM":
        Path(
            "results/cross_session/riemannian/tangent_rbf_svm/"
            "tangent_rbf_svm_cross_session_subject_results.csv"
        ),

    "Filter-Bank Riemannian + Shrinkage LDA":
        Path(
            "results/cross_session/riemannian/filterbank/lda/"
            "filterbank_riemannian_lda_cross_session_subject_results.csv"
        ),

    "Filter-Bank Riemannian + RBF-SVM":
        Path(
            "results/cross_session/riemannian/filterbank/svm/"
            "filterbank_riemannian_svm_cross_session_subject_results.csv"
        ),

    "Autoencoder + LDA":
        Path(
            "results/cross_session/autoencoder_lda/"
            "autoencoder_lda_cross_session_subject_results.csv"
        ),

    "Autoencoder + RBF-SVM":
        Path(
            "results/cross_session/autoencoder_rbf_svm/"
            "autoencoder_rbf_svm_cross_session_subject_results.csv"
        ),

    "Supervised Autoencoder + LDA":
        Path(
            "results/cross_session/supervised_autoencoder_lda/"
            "supervised_autoencoder_lda_cross_session_subject_results.csv"
        ),

    "Supervised Autoencoder + RBF-SVM":
        Path(
            "results/cross_session/supervised_autoencoder_rbf_svm/"
            "supervised_autoencoder_rbf_svm_cross_session_subject_results.csv"
        ),

    "EEGNet":
        Path(
            "results/cross_session/eegnet/"
            "eegnet_cross_session_subject_results.csv"
        ),

    "EA + CSP + Shrinkage LDA":
        Path(
            "results/cross_session/euclidean_alignment/csp/lda/"
            "ea_csp_lda_cross_session_subject_results.csv"
        ),

    "EA + CSP + RBF-SVM":
        Path(
            "results/cross_session/euclidean_alignment/csp/svm/"
            "ea_csp_svm_cross_session_subject_results.csv"
        ),

    "EA + FBCSP + Shrinkage LDA":
        Path(
            "results/cross_session/euclidean_alignment/fbcsp/lda/"
            "ea_fbcsp_lda_cross_session_subject_results.csv"
        ),

    "EA + FBCSP + RBF-SVM":
        Path(
            "results/cross_session/euclidean_alignment/fbcsp/svm/"
            "ea_fbcsp_svm_cross_session_subject_results.csv"
        ),
}


# ======================================================================
# EXPLICIT FINAL CROSS-SUBJECT SUBJECT-LEVEL SOURCES
# ======================================================================

XS_SUBJECT_RESULTS = {

    "CSP + LDA":
        Path(
            "results/cross_subject/csp_lda/"
            "csp_lda_cross_subject_subject_results.csv"
        ),

    "CSP + RBF-SVM":
        Path(
            "results/cross_subject/csp_rbf_svm/"
            "csp_rbf_svm_cross_subject_subject_results.csv"
        ),

    "Tuned CSP + LDA":
        Path(
            "results/cross_subject/tuned_csp_lda/"
            "tuned_csp_lda_cross_subject_subject_results.csv"
        ),

    "Tuned CSP + RBF-SVM":
        Path(
            "results/cross_subject/tuned_csp_rbf_svm/"
            "tuned_csp_rbf_svm_cross_subject_subject_results.csv"
        ),

    "FBCSP + LDA":
        Path(
            "results/cross_subject/fbcsp_lda/"
            "fbcsp_lda_cross_subject_subject_results.csv"
        ),

    "FBCSP + RBF-SVM":
        Path(
            "results/cross_subject/fbcsp_rbf_svm/"
            "fbcsp_rbf_svm_cross_subject_subject_results.csv"
        ),

    "Riemannian MDM":
        Path(
            "results/cross_subject/riemannian_mdm/"
            "riemannian_mdm_cross_subject_subject_results.csv"
        ),

    "Riemannian TS + Shrinkage LDA":
        Path(
            "results/cross_subject/riemannian_tangent_lda/"
            "riemannian_tangent_lda_cross_subject_subject_results.csv"
        ),

    "Riemannian TS + RBF-SVM":
        Path(
            "results/cross_subject/riemannian_tangent_rbf_svm/"
            "riemannian_tangent_rbf_svm_cross_subject_subject_results.csv"
        ),

    "Filter-Bank Riemannian + Shrinkage LDA":
        Path(
            "results/cross_subject/riemannian/filterbank/lda/"
            "filterbank_riemannian_lda_cross_subject_subject_results.csv"
        ),

    "Filter-Bank Riemannian + RBF-SVM":
        Path(
            "results/cross_subject/riemannian/filterbank/svm/"
            "filterbank_riemannian_svm_cross_subject_subject_results.csv"
        ),

    "Autoencoder + LDA":
        Path(
            "results/cross_subject/autoencoder_lda/"
            "autoencoder_lda_cross_subject_subject_results.csv"
        ),

    "Autoencoder + RBF-SVM":
        Path(
            "results/cross_subject/autoencoder_rbf_svm/"
            "autoencoder_rbf_svm_cross_subject_subject_results.csv"
        ),

    "Supervised Autoencoder + LDA":
        Path(
            "results/cross_subject/supervised_autoencoder_lda/"
            "supervised_autoencoder_lda_cross_subject_subject_results.csv"
        ),

    "Supervised Autoencoder + RBF-SVM":
        Path(
            "results/cross_subject/supervised_autoencoder_rbf_svm/"
            "supervised_autoencoder_rbf_svm_cross_subject_subject_results.csv"
        ),

    # IMPORTANT:
    # Improved EEGNet was repeated over 3 random seeds.
    # For subject-level Wilcoxon, use the nine subject means.
    "EEGNet":
        Path(
            "results/cross_subject/eegnet/"
            "eegnet_loso_improved/subject_summary.csv"
        ),

    "EA + CSP + Shrinkage LDA":
        Path(
            "results/cross_subject/euclidean_alignment/csp/lda/"
            "ea_csp_lda_cross_subject_subject_results.csv"
        ),

    "EA + CSP + RBF-SVM":
        Path(
            "results/cross_subject/euclidean_alignment/csp/svm/"
            "ea_csp_svm_cross_subject_subject_results.csv"
        ),

    "EA + FBCSP + Shrinkage LDA":
        Path(
            "results/cross_subject/euclidean_alignment/fbcsp/lda/"
            "ea_fbcsp_lda_cross_subject_subject_results.csv"
        ),

    "EA + FBCSP + RBF-SVM":
        Path(
            "results/cross_subject/euclidean_alignment/fbcsp/svm/"
            "ea_fbcsp_svm_cross_subject_subject_results.csv"
        ),
}


# ======================================================================
# HELPERS
# ======================================================================

def holm_adjust(p_values):
    """
    Holm step-down family-wise error correction.
    """
    p_values = np.asarray(
        p_values,
        dtype=float,
    )

    m = len(p_values)

    order = np.argsort(p_values)

    adjusted = np.empty(
        m,
        dtype=float,
    )

    running_max = 0.0

    for rank, idx in enumerate(order):
        factor = m - rank

        value = min(
            1.0,
            factor * p_values[idx],
        )

        running_max = max(
            running_max,
            value,
        )

        adjusted[idx] = min(
            1.0,
            running_max,
        )

    return adjusted


def find_accuracy_column(df):
    candidates = [
        "accuracy_percent",
        "mean_accuracy_percent",
        "test_accuracy_percent",
        "accuracy",
        "mean_accuracy",
        "test_accuracy",
    ]

    for column in candidates:
        if column in df.columns:
            return column

    raise KeyError(
        "Could not identify subject accuracy column. "
        f"Columns={list(df.columns)}"
    )


def normalise_accuracy(series):
    values = pd.to_numeric(
        series,
        errors="raise",
    ).astype(float)

    if values.max() <= 1.0 + 1e-8:
        values = values * 100.0

    return values.to_numpy()


def find_subject_column(df):
    candidates = [
        "subject",
        "test_subject",
    ]

    for column in candidates:
        if column in df.columns:
            return column

    raise KeyError(
        "Could not identify subject column. "
        f"Columns={list(df.columns)}"
    )


def normalise_subject(value):
    value = str(value).strip()

    # Some legacy strict-LOSO results use A01T.
    if len(value) >= 3:
        return value[:3]

    return value


def read_subject_results(path):
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    if len(df) != 9:
        raise RuntimeError(
            f"{path}: expected 9 subject rows, "
            f"found {len(df)}."
        )

    subject_column = find_subject_column(df)

    accuracy_column = find_accuracy_column(df)

    out = pd.DataFrame(
        {
            "subject": [
                normalise_subject(v)
                for v in df[subject_column]
            ],
            "accuracy_percent":
                normalise_accuracy(
                    df[accuracy_column]
                ),
        }
    )

    if out["subject"].duplicated().any():
        raise RuntimeError(
            f"{path}: duplicate subjects."
        )

    expected = {
        f"A{i:02d}"
        for i in range(1, 10)
    }

    actual = set(out["subject"])

    if actual != expected:
        raise RuntimeError(
            f"{path}: subject mismatch.\n"
            f"Expected={sorted(expected)}\n"
            f"Actual={sorted(actual)}"
        )

    return (
        out.sort_values("subject")
        .reset_index(drop=True)
    )


def read_predictions(path):
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    required = [
        "trial",
        "true_label",
        "predicted_label",
        "correct",
    ]

    for column in required:
        if column not in df.columns:
            raise RuntimeError(
                f"{path}: missing {column}"
            )

    if "subject" in df.columns:
        subject_column = "subject"

    elif "test_subject" in df.columns:
        subject_column = "test_subject"

    else:
        raise RuntimeError(
            f"{path}: no subject column."
        )

    if len(df) != 2592:
        raise RuntimeError(
            f"{path}: expected 2592 predictions, "
            f"found {len(df)}."
        )

    result = pd.DataFrame(
        {
            "subject": [
                normalise_subject(v)
                for v in df[subject_column]
            ],
            "trial":
                pd.to_numeric(
                    df["trial"],
                    errors="raise",
                ).astype(int),
            "true_label":
                df["true_label"].astype(str),
            "predicted_label":
                df["predicted_label"].astype(str),
            "correct":
                pd.to_numeric(
                    df["correct"],
                    errors="raise",
                ).astype(int),
        }
    )

    expected_subjects = {
        f"A{i:02d}"
        for i in range(1, 10)
    }

    if set(result["subject"]) != expected_subjects:
        raise RuntimeError(
            f"{path}: unexpected subject set."
        )

    counts = result.groupby(
        "subject"
    ).size()

    if not np.all(
        counts.to_numpy() == 288
    ):
        raise RuntimeError(
            f"{path}: every subject must "
            "contain 288 trials."
        )

    if result[
        ["subject", "trial"]
    ].duplicated().any():
        raise RuntimeError(
            f"{path}: duplicate subject/trial pairs."
        )

    return (
        result.sort_values(
            ["subject", "trial"]
        )
        .reset_index(drop=True)
    )


# ======================================================================
# INPUT VALIDATION
# ======================================================================

def validate_sources():
    canonical = pd.read_csv(
        CANONICAL
    )

    if len(canonical) != 20:
        raise RuntimeError(
            "Canonical table does not contain 20 methods."
        )

    canonical = canonical.set_index(
        "Method"
    )

    print()
    print("=" * 100)
    print("VALIDATING FINAL STATISTICAL INPUTS")
    print("=" * 100)

    cs_subject_data = {}
    xs_subject_data = {}
    cs_prediction_data = {}

    reference_truth = None

    for method in METHOD_ORDER:

        print()
        print(method)

        # --------------------------------------------------------------
        # CROSS-SESSION SUBJECT RESULTS
        # --------------------------------------------------------------

        cs_subject = read_subject_results(
            CS_SUBJECT_RESULTS[method]
        )

        cs_mean = float(
            cs_subject[
                "accuracy_percent"
            ].mean()
        )

        expected_cs = float(
            canonical.loc[
                method,
                "Cross-Session Accuracy",
            ]
        )

        if abs(
            cs_mean - expected_cs
        ) > 0.02:
            raise RuntimeError(
                f"{method}: Cross-Session "
                f"subject mean {cs_mean:.4f} "
                f"!= canonical {expected_cs:.4f}"
            )

        cs_subject_data[
            method
        ] = cs_subject

        # --------------------------------------------------------------
        # CROSS-SESSION PREDICTIONS
        # --------------------------------------------------------------

        predictions = read_predictions(
            CS_PREDICTIONS[method]
        )

        pred_accuracy = (
            predictions[
                "correct"
            ].mean()
            * 100.0
        )

        if abs(
            pred_accuracy - expected_cs
        ) > 0.02:
            raise RuntimeError(
                f"{method}: Cross-Session "
                f"prediction accuracy "
                f"{pred_accuracy:.4f} "
                f"!= canonical {expected_cs:.4f}"
            )

        truth = predictions[
            [
                "subject",
                "trial",
                "true_label",
            ]
        ].copy()

        if reference_truth is None:
            reference_truth = truth
        else:
            if not truth.equals(
                reference_truth
            ):
                raise RuntimeError(
                    f"{method}: prediction trial "
                    "alignment differs from reference."
                )

        cs_prediction_data[
            method
        ] = predictions

        # --------------------------------------------------------------
        # CROSS-SUBJECT SUBJECT RESULTS
        # --------------------------------------------------------------

        xs_subject = read_subject_results(
            XS_SUBJECT_RESULTS[method]
        )

        xs_mean = float(
            xs_subject[
                "accuracy_percent"
            ].mean()
        )

        expected_xs = float(
            canonical.loc[
                method,
                "Cross-Subject Accuracy",
            ]
        )

        if abs(
            xs_mean - expected_xs
        ) > 0.02:
            raise RuntimeError(
                f"{method}: Cross-Subject "
                f"subject mean {xs_mean:.4f} "
                f"!= canonical {expected_xs:.4f}"
            )

        xs_subject_data[
            method
        ] = xs_subject

        print(
            f"  CS={cs_mean:.2f}%  "
            f"XS={xs_mean:.2f}%  "
            "✅"
        )

    return (
        canonical,
        cs_prediction_data,
        cs_subject_data,
        xs_subject_data,
    )


# ======================================================================
# MCNEMAR
# ======================================================================

def compute_mcnemar(
    prediction_data,
):
    records = []

    for model_a, model_b in combinations(
        METHOD_ORDER,
        2,
    ):

        a = prediction_data[
            model_a
        ]

        b = prediction_data[
            model_b
        ]

        a_correct = (
            a["correct"].to_numpy()
            == 1
        )

        b_correct = (
            b["correct"].to_numpy()
            == 1
        )

        both_correct = int(
            np.sum(
                a_correct
                & b_correct
            )
        )

        a_correct_b_wrong = int(
            np.sum(
                a_correct
                & ~b_correct
            )
        )

        a_wrong_b_correct = int(
            np.sum(
                ~a_correct
                & b_correct
            )
        )

        both_wrong = int(
            np.sum(
                ~a_correct
                & ~b_correct
            )
        )

        discordant = (
            a_correct_b_wrong
            + a_wrong_b_correct
        )

        if discordant == 0:
            p = 1.0
        else:
            p = binomtest(
                min(
                    a_correct_b_wrong,
                    a_wrong_b_correct,
                ),
                n=discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue

        acc_a = float(
            a_correct.mean()
            * 100.0
        )

        acc_b = float(
            b_correct.mean()
            * 100.0
        )

        records.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "n_trials": len(a),
                "model_a_accuracy_percent":
                    acc_a,
                "model_b_accuracy_percent":
                    acc_b,
                "difference_pp_a_minus_b":
                    acc_a - acc_b,
                "both_correct":
                    both_correct,
                "a_correct_b_wrong":
                    a_correct_b_wrong,
                "a_wrong_b_correct":
                    a_wrong_b_correct,
                "both_wrong":
                    both_wrong,
                "discordant_pairs":
                    discordant,
                "mcnemar_p_exact":
                    float(p),
            }
        )

    df = pd.DataFrame(
        records
    )

    df[
        "holm_adjusted_p"
    ] = holm_adjust(
        df["mcnemar_p_exact"]
    )

    df[
        "significant_0_05_raw"
    ] = (
        df["mcnemar_p_exact"]
        < 0.05
    )

    df[
        "significant_0_05_holm"
    ] = (
        df["holm_adjusted_p"]
        < 0.05
    )

    return df


# ======================================================================
# WILCOXON
# ======================================================================

def compute_wilcoxon(
    subject_data,
):
    records = []

    for model_a, model_b in combinations(
        METHOD_ORDER,
        2,
    ):

        a = subject_data[
            model_a
        ].sort_values(
            "subject"
        )

        b = subject_data[
            model_b
        ].sort_values(
            "subject"
        )

        if not np.array_equal(
            a["subject"].to_numpy(),
            b["subject"].to_numpy(),
        ):
            raise RuntimeError(
                f"Subject mismatch: "
                f"{model_a} vs {model_b}"
            )

        a_values = a[
            "accuracy_percent"
        ].to_numpy()

        b_values = b[
            "accuracy_percent"
        ].to_numpy()

        differences = (
            a_values
            - b_values
        )

        if np.allclose(
            differences,
            0.0,
        ):
            statistic = 0.0
            p = 1.0
        else:
            result = wilcoxon(
                a_values,
                b_values,
                alternative="two-sided",
                zero_method="wilcox",
                correction=False,
                method="auto",
            )

            statistic = float(
                result.statistic
            )

            p = float(
                result.pvalue
            )

        records.append(
            {
                "model_a":
                    model_a,
                "model_b":
                    model_b,
                "n_subjects":
                    9,
                "model_a_mean_accuracy_percent":
                    float(
                        np.mean(
                            a_values
                        )
                    ),
                "model_b_mean_accuracy_percent":
                    float(
                        np.mean(
                            b_values
                        )
                    ),
                "mean_difference_pp_a_minus_b":
                    float(
                        np.mean(
                            differences
                        )
                    ),
                "median_difference_pp_a_minus_b":
                    float(
                        np.median(
                            differences
                        )
                    ),
                "wilcoxon_statistic":
                    statistic,
                "wilcoxon_p":
                    p,
            }
        )

    df = pd.DataFrame(
        records
    )

    df[
        "holm_adjusted_p"
    ] = holm_adjust(
        df["wilcoxon_p"]
    )

    df[
        "significant_0_05_raw"
    ] = (
        df["wilcoxon_p"]
        < 0.05
    )

    df[
        "significant_0_05_holm"
    ] = (
        df["holm_adjusted_p"]
        < 0.05
    )

    return df


# ======================================================================
# MATRICES
# ======================================================================

def build_matrix(
    df,
):
    matrix = pd.DataFrame(
        np.ones(
            (
                len(METHOD_ORDER),
                len(METHOD_ORDER),
            )
        ),
        index=METHOD_ORDER,
        columns=METHOD_ORDER,
    )

    for _, row in df.iterrows():

        a = row["model_a"]
        b = row["model_b"]

        p = float(
            row[
                "holm_adjusted_p"
            ]
        )

        matrix.loc[
            a,
            b,
        ] = p

        matrix.loc[
            b,
            a,
        ] = p

    return matrix


# ======================================================================
# WINNER COMPARISON TABLES
# ======================================================================

def winner_comparisons(
    df,
    winner,
    test_name,
):
    records = []

    for _, row in df.iterrows():

        a = row["model_a"]
        b = row["model_b"]

        if (
            a != winner
            and b != winner
        ):
            continue

        if a == winner:
            other = b
        else:
            other = a

        if test_name == "mcnemar":

            if a == winner:
                winner_acc = row[
                    "model_a_accuracy_percent"
                ]

                other_acc = row[
                    "model_b_accuracy_percent"
                ]

            else:
                winner_acc = row[
                    "model_b_accuracy_percent"
                ]

                other_acc = row[
                    "model_a_accuracy_percent"
                ]

            raw_p = row[
                "mcnemar_p_exact"
            ]

        else:

            if a == winner:
                winner_acc = row[
                    "model_a_mean_accuracy_percent"
                ]

                other_acc = row[
                    "model_b_mean_accuracy_percent"
                ]

            else:
                winner_acc = row[
                    "model_b_mean_accuracy_percent"
                ]

                other_acc = row[
                    "model_a_mean_accuracy_percent"
                ]

            raw_p = row[
                "wilcoxon_p"
            ]

        records.append(
            {
                "winner":
                    winner,
                "comparison_model":
                    other,
                "winner_accuracy_percent":
                    float(
                        winner_acc
                    ),
                "comparison_accuracy_percent":
                    float(
                        other_acc
                    ),
                "winner_advantage_pp":
                    float(
                        winner_acc
                        - other_acc
                    ),
                "raw_p":
                    float(
                        raw_p
                    ),
                "holm_adjusted_p":
                    float(
                        row[
                            "holm_adjusted_p"
                        ]
                    ),
                "significant_0_05_holm":
                    bool(
                        row[
                            "significant_0_05_holm"
                        ]
                    ),
            }
        )

    return pd.DataFrame(
        records
    ).sort_values(
        "winner_advantage_pp",
        ascending=False,
    )


# ======================================================================
# MAIN
# ======================================================================

def main():

    CS_OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    XS_OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DATA.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        canonical,
        cs_predictions,
        cs_subjects,
        xs_subjects,
    ) = validate_sources()

    # --------------------------------------------------------------
    # WINNERS
    # --------------------------------------------------------------

    best_cs = canonical[
        "Cross-Session Accuracy"
    ].idxmax()

    best_xs = canonical[
        "Cross-Subject Accuracy"
    ].idxmax()

    if (
        best_cs
        != BEST_CROSS_SESSION
    ):
        raise RuntimeError(
            f"Unexpected CS winner: "
            f"{best_cs}"
        )

    if (
        best_xs
        != BEST_CROSS_SUBJECT
    ):
        raise RuntimeError(
            f"Unexpected XS winner: "
            f"{best_xs}"
        )

    print()
    print("=" * 100)
    print("COMPUTING EXACT MCNEMAR — CROSS-SESSION")
    print("=" * 100)

    cs_mcnemar = compute_mcnemar(
        cs_predictions
    )

    cs_mcnemar_path = (
        CS_OUTPUT
        / "cross_session_mcnemar_results.csv"
    )

    cs_mcnemar.to_csv(
        cs_mcnemar_path,
        index=False,
    )

    print(
        "Pairwise comparisons:",
        len(cs_mcnemar),
    )

    print(
        "Holm-significant:",
        int(
            cs_mcnemar[
                "significant_0_05_holm"
            ].sum()
        ),
    )

    print(
        "Saved:",
        cs_mcnemar_path,
    )

    # --------------------------------------------------------------

    print()
    print("=" * 100)
    print("COMPUTING WILCOXON — CROSS-SESSION")
    print("=" * 100)

    cs_wilcoxon = compute_wilcoxon(
        cs_subjects
    )

    cs_wilcoxon_path = (
        CS_OUTPUT
        / "cross_session_wilcoxon_results.csv"
    )

    cs_wilcoxon.to_csv(
        cs_wilcoxon_path,
        index=False,
    )

    print(
        "Pairwise comparisons:",
        len(cs_wilcoxon),
    )

    print(
        "Holm-significant:",
        int(
            cs_wilcoxon[
                "significant_0_05_holm"
            ].sum()
        ),
    )

    print(
        "Saved:",
        cs_wilcoxon_path,
    )

    # --------------------------------------------------------------

    print()
    print("=" * 100)
    print("COMPUTING WILCOXON — CROSS-SUBJECT")
    print("=" * 100)

    xs_wilcoxon = compute_wilcoxon(
        xs_subjects
    )

    xs_wilcoxon_path = (
        XS_OUTPUT
        / "cross_subject_wilcoxon_results.csv"
    )

    xs_wilcoxon.to_csv(
        xs_wilcoxon_path,
        index=False,
    )

    print(
        "Pairwise comparisons:",
        len(xs_wilcoxon),
    )

    print(
        "Holm-significant:",
        int(
            xs_wilcoxon[
                "significant_0_05_holm"
            ].sum()
        ),
    )

    print(
        "Saved:",
        xs_wilcoxon_path,
    )

    # --------------------------------------------------------------
    # MATRICES
    # --------------------------------------------------------------

    cs_mcnemar_matrix = build_matrix(
        cs_mcnemar
    )

    cs_wilcoxon_matrix = build_matrix(
        cs_wilcoxon
    )

    xs_wilcoxon_matrix = build_matrix(
        xs_wilcoxon
    )

    cs_mcnemar_matrix.to_csv(
        FIGURE_DATA
        / "cross_session_mcnemar_adjusted_p_matrix.csv"
    )

    cs_wilcoxon_matrix.to_csv(
        FIGURE_DATA
        / "cross_session_wilcoxon_adjusted_p_matrix.csv"
    )

    xs_wilcoxon_matrix.to_csv(
        FIGURE_DATA
        / "cross_subject_wilcoxon_adjusted_p_matrix.csv"
    )

    # --------------------------------------------------------------
    # WINNER TABLES
    # --------------------------------------------------------------

    cs_mcnemar_best = winner_comparisons(
        cs_mcnemar,
        BEST_CROSS_SESSION,
        "mcnemar",
    )

    cs_wilcoxon_best = winner_comparisons(
        cs_wilcoxon,
        BEST_CROSS_SESSION,
        "wilcoxon",
    )

    xs_wilcoxon_best = winner_comparisons(
        xs_wilcoxon,
        BEST_CROSS_SUBJECT,
        "wilcoxon",
    )

    cs_mcnemar_best.to_csv(
        CS_OUTPUT
        / "cross_session_best_model_mcnemar.csv",
        index=False,
    )

    cs_wilcoxon_best.to_csv(
        CS_OUTPUT
        / "cross_session_best_model_wilcoxon.csv",
        index=False,
    )

    xs_wilcoxon_best.to_csv(
        XS_OUTPUT
        / "cross_subject_best_model_wilcoxon.csv",
        index=False,
    )

    print()
    print("=" * 100)
    print("FINAL WINNER COMPARISONS")
    print("=" * 100)

    print()
    print(
        "Cross-Session winner:",
        BEST_CROSS_SESSION,
    )

    print(
        "Cross-Session McNemar significant "
        "comparisons after Holm:",
        int(
            cs_mcnemar_best[
                "significant_0_05_holm"
            ].sum()
        ),
        "/",
        len(
            cs_mcnemar_best
        ),
    )

    print(
        "Cross-Session Wilcoxon significant "
        "comparisons after Holm:",
        int(
            cs_wilcoxon_best[
                "significant_0_05_holm"
            ].sum()
        ),
        "/",
        len(
            cs_wilcoxon_best
        ),
    )

    print()
    print(
        "Cross-Subject winner:",
        BEST_CROSS_SUBJECT,
    )

    print(
        "Cross-Subject Wilcoxon significant "
        "comparisons after Holm:",
        int(
            xs_wilcoxon_best[
                "significant_0_05_holm"
            ].sum()
        ),
        "/",
        len(
            xs_wilcoxon_best
        ),
    )

    print()
    print("=" * 100)
    print("✅ 20 METHODS VALIDATED AGAINST CANONICAL RESULTS")
    print("✅ 190 CROSS-SESSION MCNEMAR COMPARISONS")
    print("✅ 190 CROSS-SESSION WILCOXON COMPARISONS")
    print("✅ 190 CROSS-SUBJECT WILCOXON COMPARISONS")
    print("✅ HOLM CORRECTION APPLIED SEPARATELY TO EACH TEST FAMILY")
    print("✅ TWO PROTOCOL-SPECIFIC WINNERS VERIFIED")
    print("=" * 100)


if __name__ == "__main__":
    main()

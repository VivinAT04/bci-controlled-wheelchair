import warnings
import numpy as np
import pandas as pd
import mne

from scipy.stats import chi2
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import accuracy_score, cohen_kappa_score
from mne.decoding import CSP

from bci_wheelchair.data.loading import load_raw_gdf
from bci_wheelchair.data.preprocessing import preprocess_raw
from bci_wheelchair.models import make_csp_lda, make_fbcsp_lda


warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

SUBJECT = "A01T"


def make_tuned_csp_lda():
    return Pipeline([
        ("csp", CSP(n_components=10, reg=None, log=True, rank={"eeg": 22})),
        ("lda", LDA()),
    ])


def run_predictions(name, clf, X, y):
    print(f"Running LOOCV for {name}...")

    cv = LeaveOneOut()
    y_pred = cross_val_predict(clf, X, y, cv=cv, n_jobs=-1)

    acc = accuracy_score(y, y_pred)
    kappa = cohen_kappa_score(y, y_pred)

    print(f"{name}: Accuracy={acc * 100:.2f}%, Kappa={kappa:.3f}")

    return y_pred, acc, kappa


def mcnemar_test(y_true, pred_a, pred_b, name_a, name_b):
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
        "Significant p<0.05": p_value < 0.05,
        "Significant Bonferroni p<0.0167": p_value < (0.05 / 3),
    }


def main():
    print("\n========================================")
    print("McNemar Statistical Tests")
    print("========================================")

    raw = load_raw_gdf(f"data/raw/{SUBJECT}.gdf")

    X_base, y = preprocess_raw(raw, fmin=4, fmax=40)
    X_tuned, y_tuned = preprocess_raw(raw, fmin=8, fmax=30)
    X_fbcsp, y_fbcsp = preprocess_raw(raw, fmin=4, fmax=40)

    assert np.array_equal(y, y_tuned)
    assert np.array_equal(y, y_fbcsp)

    csp_pred, _, _ = run_predictions(
        "CSP",
        make_csp_lda(n_components=6),
        X_base,
        y,
    )

    tuned_pred, _, _ = run_predictions(
        "Tuned CSP",
        make_tuned_csp_lda(),
        X_tuned,
        y,
    )

    fbcsp_pred, _, _ = run_predictions(
        "FBCSP",
        make_fbcsp_lda(n_components=4),
        X_fbcsp,
        y,
    )

    predictions = pd.DataFrame({
        "trial": np.arange(1, len(y) + 1),
        "y_true": y,
        "csp_pred": csp_pred,
        "tuned_csp_pred": tuned_pred,
        "fbcsp_pred": fbcsp_pred,
    })

    predictions.to_csv("results/within_subject/statistical_tests/mcnemar_predictions.csv", index=False)

    results = [
        mcnemar_test(y, csp_pred, tuned_pred, "CSP", "Tuned CSP"),
        mcnemar_test(y, csp_pred, fbcsp_pred, "CSP", "FBCSP"),
        mcnemar_test(y, tuned_pred, fbcsp_pred, "Tuned CSP", "FBCSP"),
    ]

    results_df = pd.DataFrame(results)
    results_df.to_csv("results/within_subject/statistical_tests/mcnemar_results.csv", index=False)

    print("\n========================================")
    print("McNemar Test Results")
    print("========================================")
    print(results_df.to_string(index=False))

    print("\nSaved:")
    print("results/within_subject/statistical_tests/mcnemar_predictions.csv")
    print("results/within_subject/statistical_tests/mcnemar_results.csv")


if __name__ == "__main__":
    main()
"""
Preliminary classification:
A01T CSP + LDA using Leave-One-Out Cross Validation.

Run:
    python -m scripts.run_loocv_csp_a01
"""

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from mne.decoding import CSP

from bci_wheelchair.data_loading import load_subject_local


def main():

    print("Loading A01T...")
    X, y = load_subject_local("data/raw/A01T.gdf")

    print(
        f"Loaded {X.shape[0]} trials, "
        f"{X.shape[1]} EEG channels, "
        f"{X.shape[2]} samples/trial"
    )

    clf = Pipeline([
        (
            "csp",
            CSP(
                n_components=6,
                reg=None,
                log=True,
                rank={"eeg": 22}
            )
        ),
        ("lda", LDA()),
    ])

    cv = LeaveOneOut()

    print("Running LOOCV...")
    y_pred = cross_val_predict(
        clf,
        X,
        y,
        cv=cv,
        n_jobs=-1
    )

    acc = accuracy_score(y, y_pred)
    kappa = cohen_kappa_score(y, y_pred)

    print("\n--- A01T CSP + LDA LOOCV ---")
    print(f"Accuracy: {acc:.3f} ({acc*100:.1f}%)")
    print(f"Kappa: {kappa:.3f}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y, y_pred))


if __name__ == "__main__":
    main()


import mne
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

from pyriemann.estimation import Covariances
from pyriemann.classification import MDM

from bci_wheelchair.data.loading import load_raw_gdf
from bci_wheelchair.data.preprocessing import preprocess_raw


mne.set_log_level("ERROR")

SUBJECTS = [f"A{i:02d}T" for i in range(1, 10)]


def make_riemannian_mdm():
    return Pipeline([
        ("cov", Covariances(estimator="oas")),
        ("mdm", MDM(metric="riemann")),
    ])


def main():
    print("\n========================================")
    print("Riemannian Baseline: Covariance + MDM")
    print("========================================")

    all_acc = []
    all_kappa = []
    matrices = {}

    for subject in SUBJECTS:
        print(f"\nRunning {subject}...")

        raw = load_raw_gdf(f"data/raw/{subject}.gdf")
        X, y = preprocess_raw(raw, fmin=8.0, fmax=30.0, tmin=0.5, tmax=2.5)

        clf = make_riemannian_mdm()
        cv = LeaveOneOut()

        y_pred = cross_val_predict(clf, X, y, cv=cv)

        acc = accuracy_score(y, y_pred)
        kappa = cohen_kappa_score(y, y_pred)
        cm = confusion_matrix(y, y_pred)

        all_acc.append(acc)
        all_kappa.append(kappa)
        matrices[subject] = cm

        print(f"{subject} Accuracy: {acc:.3f} ({acc * 100:.1f}%)")
        print(f"{subject} Kappa:    {kappa:.3f}")

    print("\n========================================")
    print("Final Subject-wise Results")
    print("========================================")
    print(f"{'Subject':<10} {'Accuracy':<12} {'Kappa':<10}")
    print("-" * 40)

    for subject, acc, kappa in zip(SUBJECTS, all_acc, all_kappa):
        print(f"{subject:<10} {acc * 100:<12.1f} {kappa:<10.3f}")

    print("-" * 40)
    print(f"{'Mean':<10} {np.mean(all_acc) * 100:<12.1f} {np.mean(all_kappa):<10.3f}")

    print("\n========================================")
    print("Confusion Matrices")
    print("========================================")

    for subject, cm in matrices.items():
        print(f"\n{subject}")
        print(cm)


if __name__ == "__main__":
    main()
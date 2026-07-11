import mne
import numpy as np

from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

from bci_wheelchair.data_loading import load_raw_gdf
from bci_wheelchair.preprocessing import preprocess_raw
from bci_wheelchair.models import make_csp_lda, make_fbcsp_lda, make_fbcsp_svm


mne.set_log_level("ERROR")

SUBJECTS = [f"A{i:02d}" for i in range(1, 10)]


MODELS = {
    "CSP + LDA": make_csp_lda(n_components=6),
    "FBCSP + LDA": make_fbcsp_lda(n_components=4),
    "FBCSP + SVM": make_fbcsp_svm(n_components=4),
}


def evaluate_model(model_name, make_model_fn):
    print("\n========================================")
    print(f"Evaluation Dataset Protocol: {model_name}")
    print("Train: AxxT | Test: AxxE")
    print("========================================")

    accuracies = []
    kappas = []
    matrices = {}

    for subject in SUBJECTS:
        train_file = f"data/raw/{subject}T.gdf"
        test_file = f"data/raw/{subject}E.gdf"

        print(f"\nRunning {subject}: {subject}T -> {subject}E")

        raw_train = load_raw_gdf(train_file)
        raw_test = load_raw_gdf(test_file)

        X_train, y_train = preprocess_raw(raw_train, fmin=8.0, fmax=30.0, tmin=0.5, tmax=2.5)
        X_test, y_test = preprocess_raw(raw_test, fmin=8.0, fmax=30.0, tmin=0.5, tmax=2.5)

        model = make_model_fn()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        accuracies.append(acc)
        kappas.append(kappa)
        matrices[subject] = cm

        print(f"{subject} Accuracy: {acc:.3f} ({acc * 100:.1f}%)")
        print(f"{subject} Kappa:    {kappa:.3f}")

    print("\nFinal Subject-wise Results")
    print("-" * 40)
    print(f"{'Subject':<10} {'Accuracy':<12} {'Kappa':<10}")
    print("-" * 40)

    for subject, acc, kappa in zip(SUBJECTS, accuracies, kappas):
        print(f"{subject:<10} {acc * 100:<12.1f} {kappa:<10.3f}")

    print("-" * 40)
    print(f"{'Mean':<10} {np.mean(accuracies) * 100:<12.1f} {np.mean(kappas):<10.3f}")

    print("\nConfusion Matrices")
    print("=" * 40)
    for subject, cm in matrices.items():
        print(f"\n{subject}")
        print(cm)


def main():
    evaluate_model("CSP + LDA", lambda: make_csp_lda(n_components=6))
    evaluate_model("FBCSP + LDA", lambda: make_fbcsp_lda(n_components=4))
    evaluate_model("FBCSP + SVM", lambda: make_fbcsp_svm(n_components=4))


if __name__ == "__main__":
    main()
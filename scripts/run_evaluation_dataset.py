import mne
import numpy as np

from scipy.io import loadmat
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

from bci_wheelchair.data_loading import load_raw_gdf
from bci_wheelchair.preprocessing import preprocess_raw
from bci_wheelchair.models import make_csp_lda, make_fbcsp_lda, make_fbcsp_svm


mne.set_log_level("ERROR")

SUBJECTS = [f"A{i:02d}" for i in range(1, 10)]

LABEL_MAP = {
    1: "left_hand",
    2: "right_hand",
    3: "feet",
    4: "tongue",
}


def preprocess_evaluation_raw(
    raw,
    fmin=8.0,
    fmax=30.0,
    tmin=0.5,
    tmax=2.5,
):
    """
    Preprocess an evaluation GDF file.

    Evaluation files use event code 783 for every trial, while the true
    class labels are stored separately in the corresponding MAT file.
    """
    raw = raw.copy()

    raw.drop_channels(
        ["EOG-left", "EOG-central", "EOG-right"],
        on_missing="ignore",
    )

    raw.filter(
        fmin,
        fmax,
        fir_design="firwin",
        verbose=False,
    )

    events, _ = mne.events_from_annotations(
        raw,
        event_id={"783": 1},
        verbose=False,
    )

    epochs = mne.Epochs(
        raw,
        events,
        event_id={"783": 1},
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose=False,
    )

    return epochs.get_data()


def load_evaluation_labels(label_file):
    """Load and convert evaluation labels from the official MAT file."""
    mat_data = loadmat(label_file)

    numeric_labels = mat_data["classlabel"].reshape(-1)

    return np.array(
        [LABEL_MAP[int(label)] for label in numeric_labels]
    )


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
        label_file = f"data/labels/{subject}E.mat"

        print(f"\nRunning {subject}: {subject}T -> {subject}E")

        raw_train = load_raw_gdf(train_file)
        raw_test = load_raw_gdf(test_file)

        X_train, y_train = preprocess_raw(
            raw_train,
            fmin=8.0,
            fmax=30.0,
            tmin=0.5,
            tmax=2.5,
        )

        X_test = preprocess_evaluation_raw(
            raw_test,
            fmin=8.0,
            fmax=30.0,
            tmin=0.5,
            tmax=2.5,
        )

        y_test = load_evaluation_labels(label_file)

        if len(X_test) != len(y_test):
            raise ValueError(
                f"{subject}: number of evaluation epochs "
                f"({len(X_test)}) does not match number of labels "
                f"({len(y_test)})."
            )

        model = make_model_fn()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)
        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=["left_hand", "right_hand", "feet", "tongue"],
        )

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
    print(
        f"{'Mean':<10} "
        f"{np.mean(accuracies) * 100:<12.1f} "
        f"{np.mean(kappas):<10.3f}"
    )

    print("\nConfusion Matrices")
    print("=" * 40)

    for subject, cm in matrices.items():
        print(f"\n{subject}")
        print(cm)


def main():
    evaluate_model(
        "CSP + LDA",
        lambda: make_csp_lda(n_components=6),
    )

    evaluate_model(
        "FBCSP + LDA",
        lambda: make_fbcsp_lda(n_components=4),
    )

    evaluate_model(
        "FBCSP + SVM",
        lambda: make_fbcsp_svm(n_components=4),
    )


if __name__ == "__main__":
    main()

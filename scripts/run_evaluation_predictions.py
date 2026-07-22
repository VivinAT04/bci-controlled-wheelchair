import mne
import numpy as np
import pandas as pd

from scipy.io import loadmat

from bci_wheelchair.data_loading import load_raw_gdf
from bci_wheelchair.preprocessing import preprocess_raw
from bci_wheelchair.models import make_fbcsp_svm


mne.set_log_level("ERROR")

SUBJECTS = [f"A{i:02d}" for i in range(1, 10)]

LABEL_MAP = {
    1: "left_hand",
    2: "right_hand",
    3: "feet",
    4: "tongue",
}


def preprocess_eval_raw(
    raw,
    fmin=8.0,
    fmax=30.0,
    tmin=0.5,
    tmax=2.5,
):
    """Preprocess one evaluation GDF file using event code 783."""
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


def load_evaluation_labels(label_path):
    """Load true evaluation labels from the official MAT file."""
    mat_data = loadmat(label_path)
    numeric_labels = mat_data["classlabel"].reshape(-1)

    return np.array([
        LABEL_MAP[int(label)]
        for label in numeric_labels
    ])


def main():
    all_predictions = []

    for subject in SUBJECTS:
        print(
            f"\nRunning {subject}: "
            f"train {subject}T -> predict {subject}E"
        )

        train_path = f"data/raw/{subject}T.gdf"
        eval_path = f"data/raw/{subject}E.gdf"
        label_path = f"data/labels/{subject}E.mat"

        raw_train = load_raw_gdf(train_path)
        raw_eval = load_raw_gdf(eval_path)

        X_train, y_train = preprocess_raw(
            raw_train,
            fmin=8.0,
            fmax=30.0,
            tmin=0.5,
            tmax=2.5,
        )

        X_eval = preprocess_eval_raw(
            raw_eval,
            fmin=8.0,
            fmax=30.0,
            tmin=0.5,
            tmax=2.5,
        )

        y_true = load_evaluation_labels(label_path)

        if len(X_eval) != len(y_true):
            raise ValueError(
                f"{subject}: {len(X_eval)} epochs but "
                f"{len(y_true)} true labels."
            )

        model = make_fbcsp_svm(n_components=4)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_eval)

        for trial, (true_label, predicted_label) in enumerate(
            zip(y_true, y_pred),
            start=1,
        ):
            all_predictions.append({
                "subject": subject,
                "trial": trial,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "correct": int(true_label == predicted_label),
            })

        subject_accuracy = np.mean(y_true == y_pred)

        print(
            f"{subject}: generated {len(y_pred)} predictions "
            f"with accuracy {subject_accuracy * 100:.1f}%"
        )

    dataframe = pd.DataFrame(all_predictions)

    output_path = "results/evaluation_predictions.csv"
    dataframe.to_csv(output_path, index=False)

    print("\n========================================")
    print("Evaluation Predictions Complete")
    print("========================================")
    print(f"Total predictions: {len(dataframe)}")
    print(
        f"Overall accuracy: "
        f"{dataframe['correct'].mean() * 100:.1f}%"
    )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()

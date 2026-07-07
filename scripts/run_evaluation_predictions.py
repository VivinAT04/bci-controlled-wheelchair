import mne
import pandas as pd

from bci_wheelchair.data_loading import load_raw_gdf
from bci_wheelchair.preprocessing import preprocess_raw
from bci_wheelchair.models import make_fbcsp_lda

mne.set_log_level("ERROR")

SUBJECTS = [f"A{i:02d}" for i in range(1, 10)]

def preprocess_eval_raw(raw, fmin=8.0, fmax=30.0, tmin=0.5, tmax=2.5):
    raw = raw.copy()
    raw.drop_channels(["EOG-left", "EOG-central", "EOG-right"], on_missing="ignore")
    raw.filter(fmin, fmax, fir_design="firwin", verbose=False)

    events, _ = mne.events_from_annotations(raw, event_id={"783": 1}, verbose=False)

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

def main():
    all_predictions = []

    for subject in SUBJECTS:
        print(f"\nRunning {subject}: train {subject}T -> predict {subject}E")

        raw_train = load_raw_gdf(f"data/raw/{subject}T.gdf")
        raw_eval = load_raw_gdf(f"data/raw/{subject}E.gdf")

        X_train, y_train = preprocess_raw(raw_train, fmin=8.0, fmax=30.0, tmin=0.5, tmax=2.5)
        X_eval = preprocess_eval_raw(raw_eval, fmin=8.0, fmax=30.0, tmin=0.5, tmax=2.5)

        model = make_fbcsp_lda(n_components=4)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_eval)

        for i, pred in enumerate(y_pred, start=1):
            all_predictions.append({
                "subject": subject,
                "trial": i,
                "predicted_label": pred,
            })

        print(f"{subject}: generated {len(y_pred)} predictions")

    df = pd.DataFrame(all_predictions)
    df.to_csv("results/evaluation_predictions.csv", index=False)

    print("\nSaved: results/evaluation_predictions.csv")

if __name__ == "__main__":
    main()
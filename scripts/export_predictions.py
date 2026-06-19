"""
Export classifier predictions with confidence scores to CSV, for the
MATLAB wheelchair simulation to consume.

Run from the project root:
    python3 -m scripts.export_predictions
"""

import csv

from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from bci_wheelchair.commands import CLASS_TO_COMMAND
from bci_wheelchair.data_loading import load_raw_gdf
from bci_wheelchair.preprocessing import preprocess_raw


def main(subject_path="data/raw/A01T.gdf", out_path="results/predicted_commands.csv"):
    print(f"Loading {subject_path}...")

    raw = load_raw_gdf(subject_path)
    X, y = preprocess_raw(raw)

    clf = Pipeline([
        ("csp", CSP(n_components=6, reg=None, log=True)),
        ("lda", LDA()),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("Running cross-validated predictions...")
    y_pred = cross_val_predict(clf, X, y, cv=cv)
    y_proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")
    confidence = y_proba.max(axis=1)

    print(f"Writing {out_path}...")

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trial", "true_class", "predicted_class", "command", "confidence"])

        for i, (true_c, pred_c, conf) in enumerate(zip(y, y_pred, confidence)):
            writer.writerow([
                i + 1,
                true_c,
                pred_c,
                CLASS_TO_COMMAND[pred_c],
                f"{conf:.3f}",
            ])

    print(f"Done. {len(y)} commands exported.")


if __name__ == "__main__":
    main()
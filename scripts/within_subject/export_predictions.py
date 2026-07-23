"""
Export FBCSP + LDA classifier predictions with confidence scores to CSV, for the
MATLAB wheelchair simulation to consume.

Run from the project root:
    python3 -m scripts.within_subject.export_predictions
"""

import csv

from sklearn.model_selection import StratifiedKFold, cross_val_predict

from bci_wheelchair.commands import CLASS_TO_COMMAND
from bci_wheelchair.data.loading import load_raw_gdf
from bci_wheelchair.models import make_fbcsp_lda
from bci_wheelchair.data.preprocessing import preprocess_raw


def main(subject_path="data/raw/A01T.gdf", out_path="results/within_subject/predictions/predicted_commands.csv"):
    print(f"Loading {subject_path}...")

    raw = load_raw_gdf(subject_path)
    X, y = preprocess_raw(raw)

    print("Building FBCSP + LDA classifier...")
    clf = make_fbcsp_lda(n_components=4)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("Running cross-validated FBCSP predictions...")
    y_pred = cross_val_predict(clf, X, y, cv=cv)
    y_proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")
    confidence = y_proba.max(axis=1)

    print(f"Writing {out_path}...")

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "trial",
            "true_class",
            "predicted_class",
            "command",
            "confidence",
            "model",
        ])

        for i, (true_c, pred_c, conf) in enumerate(zip(y, y_pred, confidence)):
            writer.writerow([
                i + 1,
                true_c,
                pred_c,
                CLASS_TO_COMMAND[pred_c],
                f"{conf:.3f}",
                "FBCSP_LDA",
            ])

    print(f"Done. {len(y)} FBCSP commands exported.")


if __name__ == "__main__":
    main()
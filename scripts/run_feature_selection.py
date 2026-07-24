"""Run FBCSP feature-selection experiments."""

from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from bci_wheelchair.data_loading import load_raw_gdf
from bci_wheelchair.preprocessing import preprocess_raw
from bci_wheelchair.models import (
    make_fbcsp_lda,
    make_fbcsp_feature_selected_lda,
)

SUBJECT = "A01T"
GDF_PATH = f"data/raw/{SUBJECT}.gdf"

PERCENTILES = [90, 80, 70, 60, 50]


def evaluate_model(name, model, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    y_pred = cross_val_predict(model, X, y, cv=cv)

    acc = accuracy_score(y, y_pred)
    kappa = cohen_kappa_score(y, y_pred)

    print(f"{name}: Accuracy={acc * 100:.2f}%, Kappa={kappa:.3f}")


def main():
    print("=" * 50)
    print("FBCSP Feature Selection Experiment")
    print("=" * 50)

    raw = load_raw_gdf(GDF_PATH)
    X, y = preprocess_raw(raw)

    print(f"Loaded subject: {SUBJECT}")
    print(f"Trials: {X.shape[0]}")
    print(f"Channels: {X.shape[1]}")
    print(f"Samples: {X.shape[2]}")
    print()

    print("Baseline:")
    evaluate_model("FBCSP baseline", make_fbcsp_lda(), X, y)

    print()
    print("Feature-selected models:")

    for percentile in PERCENTILES:
        model = make_fbcsp_feature_selected_lda(percentile=percentile)
        evaluate_model(f"FBCSP + top {percentile}% features", model, X, y)


if __name__ == "__main__":
    main()
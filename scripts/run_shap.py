import numpy as np
import mne
from sklearn.pipeline import Pipeline

from bci_wheelchair.explainability.shap_explainer import SHAPExplainer
from bci_wheelchair.preprocessing import load_data
from bci_wheelchair.features import extract_csp_features
from bci_wheelchair.model import get_model


def main():

    # -------------------------
    # 1. Load EEG data
    # -------------------------
    raw = load_data()

    print(f"TOTAL EVENTS: {len(raw.annotations)}")

    # FIX 1: prevent duplicate event crash
    events, event_id = mne.events_from_annotations(
        raw,
        event_repeated="merge"
    )

    # -------------------------
    # 2. Create epochs safely
    # -------------------------
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=0,
        tmax=4,
        baseline=None,
        preload=True,
        event_repeated="merge"
    )

    labels = epochs.events[:, -1]
    print("\nUNIQUE LABELS:", np.unique(labels))

    # -------------------------
    # 3. Feature extraction (CSP)
    # -------------------------
    X, y = extract_csp_features(epochs)

    print(f"\nCSP FEATURE SHAPE: {X.shape}")

    # FIX 2: ensure correct shape for SHAP
    X = np.array(X)
    if X.ndim != 2:
        raise ValueError(f"Invalid feature shape: {X.shape}")

    # -------------------------
    # 4. Train model pipeline
    # -------------------------
    model = get_model()
    model.fit(X, y)

    # -------------------------
    # 5. SHAP
    # -------------------------
    explainer = SHAPExplainer(model)
    explainer.fit(X)

    print("\nGenerating SHAP summary plot...")
    explainer.summary_plot(X)

    print("\nGenerating SHAP force plot...")
    explainer.force_plot(X[0])


if __name__ == "__main__":
    main()
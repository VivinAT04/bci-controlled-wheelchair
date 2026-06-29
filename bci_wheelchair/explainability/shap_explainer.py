import shap
import numpy as np


class SHAPExplainer:
    def __init__(self, model):
        self.model = model
        self.explainer = None
        self.shap_values = None
        self.X_background = None

    def fit(self, X):
        """
        Fit SHAP KernelExplainer safely for sklearn pipeline (CSP + classifier)
        """

        # FIX 1: ensure 2D input
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # FIX 2: limit background for speed
        max_samples = min(100, X.shape[0])
        background = shap.sample(X, max_samples)

        # IMPORTANT FIX: wrap predict_proba safely
        def model_predict(data):
            data = np.array(data)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            return self.model.predict_proba(data)

        self.explainer = shap.KernelExplainer(model_predict, background)
        self.X_background = background

        return self

    def explain(self, X):
        """
        Compute SHAP values safely
        """
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        self.shap_values = self.explainer.shap_values(X)
        return self.shap_values

    def summary_plot(self, X):
        """
        FIXED SHAP API (v0.20+)
        """
        if self.shap_values is None:
            self.explain(X)

        shap.summary_plot(self.shap_values, X)

    def force_plot(self, X_instance, class_idx=0):
        """
        FIXED force plot for SHAP v0.20+
        """

        if self.shap_values is None:
            self.explain(np.array([X_instance]))

        # FIX: correct API usage
        shap.plots.force(
            self.explainer.expected_value[class_idx],
            self.shap_values[class_idx][0]
        )
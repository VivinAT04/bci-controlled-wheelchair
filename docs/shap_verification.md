# SHAP Implementation Verification

The SHAP implementation was checked after the preliminary experiments.

SHAP was not applied directly to raw EEG time-series data. Instead, EEG epochs were first transformed into CSP or FBCSP feature vectors. The trained classifier output was then explained using SHAP KernelExplainer.

For the FBCSP + LDA model, the pipeline was:
Raw EEG epoch -> band-pass filtering -> FBCSP feature extraction -> LDA classifier -> SHAP explanation of LDA probability output.

For the CSP + RBF-SVM model, the pipeline was:
Raw EEG epoch -> CSP feature extraction -> RBF-SVM classifier -> SHAP explanation of SVM probability output.

The SHAP background data was sampled from the training feature set, while the explained samples were taken from the held-out test feature set. This avoids using the explained test samples as the SHAP background.

The explanations therefore show the importance of CSP/FBCSP components, not raw EEG channels. For FBCSP, feature names were constructed as frequency-band CSP components, for example 20-24Hz_CSP1.

For multiclass classification, SHAP values were aggregated across classes to obtain global feature importance. Mean absolute SHAP values were used to rank features.

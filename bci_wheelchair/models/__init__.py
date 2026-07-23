"""Reusable EEG models and classifier pipeline constructors."""

from .csp import (
    make_csp_feature_selected_lda,
    make_csp_feature_selected_svm,
    make_csp_lda,
    make_csp_svm,
)
from .eegnet import (
    EEGNet,
    initialise_eegnet,
    make_eegnet,
)
from .fbcsp import (
    DEFAULT_BANDS,
    FilterBankCSP,
    RegularizedFilterBankCSP,
    make_fbcsp_feature_selected_lda,
    make_fbcsp_feature_selected_svm,
    make_fbcsp_lda,
    make_fbcsp_svm,
)
from .riemannian import (
    make_riemannian_mdm,
    make_riemannian_mdm_pipeline,
    make_tangent_lda,
    make_tangent_lda_pipeline,
    make_tangent_svm,
    make_tangent_svm_pipeline,
)

__all__ = [
    "DEFAULT_BANDS",
    "EEGNet",
    "FilterBankCSP",
    "RegularizedFilterBankCSP",
    "initialise_eegnet",
    "make_csp_feature_selected_lda",
    "make_csp_feature_selected_svm",
    "make_csp_lda",
    "make_csp_svm",
    "make_eegnet",
    "make_fbcsp_feature_selected_lda",
    "make_fbcsp_feature_selected_svm",
    "make_fbcsp_lda",
    "make_fbcsp_svm",
    "make_riemannian_mdm",
    "make_riemannian_mdm_pipeline",
    "make_tangent_lda",
    "make_tangent_lda_pipeline",
    "make_tangent_svm",
    "make_tangent_svm_pipeline",
]

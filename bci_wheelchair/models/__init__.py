"""Reusable EEG models and classifier pipeline constructors."""

from .csp import (
    make_csp_transformer,
    make_csp_feature_selected_lda,
    make_csp_feature_selected_svm,
    make_csp_lda,
    make_csp_svm,
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


def __getattr__(name):
    """
    Lazily import optional/heavier model families.

    CSP and FBCSP remain immediately available.

    Riemannian models load pyRiemann only when requested.

    EEGNet loads PyTorch only when requested.
    """

    if name in {
        "make_riemannian_mdm",
        "make_riemannian_mdm_pipeline",
        "make_tangent_lda",
        "make_tangent_lda_pipeline",
        "make_tangent_svm",
        "make_tangent_svm_pipeline",
    }:
        from .riemannian import (
            make_riemannian_mdm,
            make_riemannian_mdm_pipeline,
            make_tangent_lda,
            make_tangent_lda_pipeline,
            make_tangent_svm,
            make_tangent_svm_pipeline,
        )

        exports = {
            "make_riemannian_mdm": make_riemannian_mdm,
            "make_riemannian_mdm_pipeline": make_riemannian_mdm_pipeline,
            "make_tangent_lda": make_tangent_lda,
            "make_tangent_lda_pipeline": make_tangent_lda_pipeline,
            "make_tangent_svm": make_tangent_svm,
            "make_tangent_svm_pipeline": make_tangent_svm_pipeline,
        }

        return exports[name]

    if name in {
        "EEGNet",
        "initialise_eegnet",
        "make_eegnet",
    }:
        from .eegnet import (
            EEGNet,
            initialise_eegnet,
            make_eegnet,
        )

        exports = {
            "EEGNet": EEGNet,
            "initialise_eegnet": initialise_eegnet,
            "make_eegnet": make_eegnet,
        }

        return exports[name]

    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
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
    "make_csp_transformer",
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

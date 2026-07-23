"""Reusable EEG feature extraction components."""

from .csp import make_csp_transformer
from .fbcsp import (
    DEFAULT_BANDS,
    FilterBankCSP,
    RegularizedFilterBankCSP,
)

__all__ = [
    "DEFAULT_BANDS",
    "FilterBankCSP",
    "RegularizedFilterBankCSP",
    "make_csp_transformer",
]

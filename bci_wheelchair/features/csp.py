"""Common Spatial Pattern feature extraction utilities."""

from __future__ import annotations

from mne.decoding import CSP


def make_csp_transformer(
    n_components: int = 6,
    regularization=None,
    log: bool = True,
) -> CSP:
    """Create a configured MNE Common Spatial Pattern transformer."""
    return CSP(
        n_components=n_components,
        reg=regularization,
        log=log,
    )


__all__ = [
    "make_csp_transformer",
]

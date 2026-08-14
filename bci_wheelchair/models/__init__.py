"""Model package for the BCI wheelchair project.

Deep-learning models are imported lazily so classical CSP/FBCSP
experiments do not unnecessarily load PyTorch.
"""

__all__ = [
    "EEGNet",
    "EEGNetConfig",
    "initialise_eegnet",
]


def __getattr__(name):
    """Load EEGNet components only when explicitly requested."""
    if name in __all__:
        from .eegnet import (
            EEGNet,
            EEGNetConfig,
            initialise_eegnet,
        )

        exports = {
            "EEGNet": EEGNet,
            "EEGNetConfig": EEGNetConfig,
            "initialise_eegnet": initialise_eegnet,
        }

        return exports[name]

    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )

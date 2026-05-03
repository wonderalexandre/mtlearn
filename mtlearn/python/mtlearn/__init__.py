"""Python interface for mtlearn."""

try:
    # Torch must be imported before the native module so the compiled bindings
    # can use its Tensor converters.
    import torch  # noqa: F401
except ImportError as exc:  # pragma: no cover - explicit runtime failure
    raise ImportError("mtlearn requires torch to load its compiled bindings") from exc


from . import morphology
from . import layers
from . import datasets as datasets
from . import data as data
from ._native import load_bindings

_bindings = load_bindings()

WITH_TORCH = getattr(_bindings, "WITH_TORCH", False)

ConnectedFilterPreprocessingTreeTensors = getattr(_bindings, "ConnectedFilterPreprocessingTreeTensors", None)
ConnectedFilterPreprocessingTreeTraversal = getattr(_bindings, "ConnectedFilterPreprocessingTreeTraversal", None)


__all__ = [
    "WITH_TORCH",
    "data",
    "datasets",
    "layers",
    "morphology",
    "ConnectedFilterPreprocessingTreeTensors",
    "ConnectedFilterPreprocessingTreeTraversal",
]

__version__ = "1.0.4"

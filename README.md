# mtlearn

`mtlearn` is a C++/Python research library for learnable connected operators
based on morphological trees.

The library explores a simple idea: connected morphology can become a structural
prior for deep neural networks. Instead of processing images only through local
pixel-wise operations, connected operators reason over components, regions,
shape, contrast, and hierarchy. This makes them naturally interpretable and
well-suited for tasks where structure matters.

Classical connected filters are powerful, but they usually depend on hard
keep/discard decisions and manually selected attribute thresholds. This limits
their integration into end-to-end trainable neural architectures.

`mtlearn` provides a stable implementation platform for this research direction.
It currently includes Connected Filter Preprocessing (CFP), and is intended to
grow toward trainable connected-operator layers, differentiable or learnable
attribute criteria, self-dual tree representations, intermediate network
insertions, and scalable implementations.

## Main Features

- **Connected Filter Preprocessing (CFP):** the current main model, available as
  `mtlearn.layers.ConnectedFilterPreprocessingLayer`. CFP replaces hard
  connected-filter decisions with a differentiable sigmoid gate over normalized
  tree-node attributes.

- **Stable morphology interface:** `mtlearn.morphology` builds max-trees,
  min-trees, and tree-of-shapes through a backend-independent API.

- **Trainable connected morphology:** designed as an implementation platform for
  connected morphology as a learnable structural prior in deep neural networks.

- **Research-ready validation:** includes C++ tests, Python tests, gradient
  checks, reference implementations, notebook validations, and public dataset
  download helpers.

## Install

From PyPI:

```bash
pip install mtlearn
```

See [docs/installation.md](docs/installation.md) for source builds, CMake
configuration, public API notes, validation, and release checks.

## Quick Start

Build a morphology tree and compute attributes:

```python
import numpy as np
from mtlearn import morphology

image = np.array([[1, 2], [3, 4]], dtype=np.uint8)
tree = morphology.create_max_tree(image)

_, attributes = morphology.compute_attributes(
    tree,
    [morphology.AttributeType.AREA, morphology.AttributeType.COMPACTNESS],
)

print(attributes.shape)
```

Create a CFP layer and run a forward pass:

```python
import torch
from mtlearn import morphology
from mtlearn.layers import ConnectedFilterPreprocessingLayer

cfp_layer = ConnectedFilterPreprocessingLayer(
    in_channels=1,
    attributes_spec=[(
        morphology.AttributeType.AREA,
        morphology.AttributeType.CIRCULARITY,
    )],
    tree_type="max-tree",
    device="cpu",
)

x = torch.tensor([[[[1, 2], [3, 4]]]], dtype=torch.float32)
y = cfp_layer(x)

assert y.shape == x.shape
```

## Implementation Notes

`ConnectedFilterPreprocessingLayer` is the recommended implementation for new
CFP experiments.

Tensor operations, trainable parameters, and cached attributes can live on CUDA
when `device="cuda"`. Morphology-tree construction is still performed by the
C++ backend on CPU.

The main implementation uses an implicit Jacobian formulation. The dense
region-pixel matrix is not materialized during normal training; tree-ordering
metadata is used to perform the equivalent reconstruction and backward
accumulation more compactly. This reduces memory pressure compared with
explicit region-pixel Jacobian construction.

Reference implementations based on explicit Jacobians and CPU tree traversals
remain available for gradient checks, comparisons, and debugging.

## Backend Notes

`mtlearn` uses a C++ morphology backend internally through `mtlearn::morphology`.
User code should interact with morphology through the public Python facade
`mtlearn.morphology`, rather than depending on backend-specific APIs.

The current backend is `MorphologicalAttributeFilters` / `mmcfilters v1.0.0`,
but the top-level Python package `mmcfilters` is not required as a runtime
dependency of `mtlearn`.

## Current Scope

`mtlearn` is currently a research-oriented library. The public API is intended
to remain stable where possible, but some components may evolve as the project
moves toward broader trainable connected-operator layers.

The current implementation supports:

- max-tree and min-tree CFP workflows through `mtlearn.morphology`;
- multi-attribute groups;
- dataset-level attribute normalization;
- cached preprocessing for indexed training datasets;
- PyTorch forward and backward for CFP parameters on CPU or CUDA tensors;
- gradcheck notebooks and CTest integration.

The current CFP layer does not yet represent the full research agenda of
`mtlearn`. It is the first validated member of a broader planned family of
trainable connected-operator layers.

## Citation

If you use the CFP layer in your work, please cite:

> Wonder A. L. Alves, Lucas de P. O. Santos, Ronaldo F. Hashimoto, Nicolas Passat, Anderson H. R. Souza, Dennis J. Silva, Yukiko Kenmochi. **A trainable connected filter preprocessing layer based on component trees.** International Conference on Pattern Recognition (ICPR), 2026, Lyon, France. ⟨[hal-05575141](https://hal.science/hal-05575141/)⟩

# Installation, Public API, and Development

This page documents installation paths, public APIs, source builds, tests,
notebook validation, and release checks for `mtlearn`.

## Requirements

For the prebuilt Python package, a recent Python environment with `pip` is
usually sufficient.

For source builds, you need a working native build environment, including:

- Python;
- a C++ compiler supported by CMake;
- CMake;
- PyTorch;
- pybind11;
- scikit-build-core.

The project uses a native `_mtlearn` extension, so source builds require a C++
build toolchain. Python bindings currently expose Torch tensors, so building the
Python extension also requires Torch support.

## Install From PyPI

```bash
pip install mtlearn
```

For notebooks:

```bash
pip install "mtlearn[notebooks]"
```

## Source Checkout

Clone with submodules:

```bash
git clone --recurse-submodules https://github.com/wonderalexandre/mtlearn.git
cd mtlearn
```

If the repository was cloned without submodules:

```bash
git submodule update --init --recursive
```

The current backend source is expected at:

```text
external/MorphologicalAttributeFilters
```

## Editable Install

```bash
pip install build scikit-build-core pybind11 torch
pip install -e .
```

For notebooks from a source checkout:

```bash
pip install -e ".[notebooks]"
```

## Wheel Build

```bash
pip install build scikit-build-core pybind11 torch
python -m build --wheel
python -m pip install dist/mtlearn-*.whl
```

The Python package uses the native `_mtlearn` extension. The top-level
`mmcfilters` Python package is not a runtime dependency of `mtlearn`.

## Public Python API

High-level modules:

- `mtlearn.morphology`: tree construction, attributes, and attribute filters;
- `mtlearn.layers`: trainable connected-layer implementations;
- `mtlearn.datasets`: dataset classes used by examples and notebooks;
- `mtlearn.data`: public dataset registry and download helpers.

Low-level binding helpers such as
`mtlearn.ConnectedFilterPreprocessingTreeTensors` and
`mtlearn.ConnectedFilterPreprocessingTreeTraversal` remain available for gradchecks,
reference implementations, and debugging. These helpers are not the recommended
entry point for new user code; new code should prefer the high-level modules
above.

### Morphology Facade

```python
from mtlearn import morphology

tree = morphology.create_max_tree(image)
tree = morphology.create_min_tree(image)
tree = morphology.create_tree_of_shapes(image)
tree = morphology.build_tree(image, "max-tree")
```

The public tree aliases are:

- `morphology.Tree`
- `morphology.WeightedTree`
- `morphology.WeightedMorphologicalTree`

Prefer `morphology.Tree` or `morphology.WeightedTree` in new code. The
`WeightedMorphologicalTree` name remains available for backend-oriented tests
and compatibility with current bindings.

Attributes and filters:

```python
Type = morphology.AttributeType
Group = morphology.AttributeGroup
Space = morphology.NodeIdSpace

_, attributes = morphology.compute_attributes(
    tree,
    [Type.AREA, Type.COMPACTNESS],
)

single = morphology.compute_single_attribute(tree, Type.AREA)
attribute_filter = morphology.create_attribute_filter(tree)

area_description = morphology.describe_attribute(Type.AREA)
all_descriptions = morphology.describe_all_attributes()
```

### Layers

The main public layer is:

```python
from mtlearn.layers import ConnectedFilterPreprocessingLayer
```

Alias:

```python
from mtlearn.layers import CFPLayer
```

Reference implementations:

- `ConnectedFilterPreprocessingLayerWithExplicitJacobian`
- `ConnectedFilterPreprocessingLayerWithCPUTreeTraversal`

Autograd functions are exported for tests and research notebooks, but most user
code should instantiate the layer classes instead.

## Public C++ API

The public C++ weighted-tree type is:

```cpp
mtlearn::morphology::WeightedTree
```

It wraps the current backend so C++ consumers do not depend directly on
`mmcfilters::WeightedMorphologicalTree`. The public header
`mtlearn/morphology.hpp` owns the public morphology enums and does not include
`mmcfilters` headers.

Installed consumers should link only against:

```cmake
find_package(mtlearn CONFIG REQUIRED)
target_link_libraries(my_target PRIVATE mtlearn::core)
```

## Development Builds

### Minimum C++-Only Build

```bash
cmake -S . -B build-cpp \
      -DMTLEARN_BUILD_PYTHON=OFF \
      -DMTLEARN_WITH_TORCH=OFF
cmake --build build-cpp
```

### C++/Python Test Build

```bash
cmake -S . -B build \
      -DMTLEARN_BUILD_TESTS=ON \
      -DMTLEARN_BUILD_PYTHON=ON \
      -DMTLEARN_WITH_TORCH=ON \
      -DMTLEARN_ENABLE_EMBED=OFF \
      -DPYTHON_EXECUTABLE=$(python -c "import sys; print(sys.executable)") \
      -DCMAKE_PREFIX_PATH="$(python -c 'import torch, pybind11; print(torch.utils.cmake_prefix_path + ";" + pybind11.get_cmake_dir())')"
cmake --build build
ctest --test-dir build --output-on-failure
```

The `CMAKE_PREFIX_PATH` expression locates both LibTorch and pybind11 from the
active Python environment.

`MTLEARN_ENABLE_EMBED=ON` activates the embedded-interpreter path in
`mtl_interpreter_test`. Before enabling it, verify that the selected environment
can import PyTorch:

```bash
python -c "import torch"
```

### CMake Options

- `MTLEARN_BUILD_PYTHON`: build the `_mtlearn` pybind11 extension. Default: `ON`.
- `MTLEARN_BUILD_TESTS`: build and register tests. Default: `OFF`.
- `MTLEARN_WITH_TORCH`: enable LibTorch-dependent code. Default: `ON`.
- `MTLEARN_ENABLE_EMBED`: enable embedded Python interpreter test behaviour.
  Default: `OFF`.
- `MTLEARN_ENABLE_ASSERTS`: keep runtime assertions enabled in core C++ code.
  Default: `OFF`.

`MTLEARN_BUILD_PYTHON=ON` currently requires `MTLEARN_WITH_TORCH=ON` because the
bindings expose Torch tensors.

## Validation

### Direct Python Tests

```bash
pip install -e ".[test]"
PYTHONPATH=mtlearn/python:build/mtlearn/bindings python -m pytest -q -m "not gradcheck" mtlearn/tests/python
PYTHONPATH=mtlearn/python:build/mtlearn/bindings python -m pytest -q -m gradcheck mtlearn/tests/python
```

Whitespace and syntax checks:

```bash
python -m compileall -q mtlearn/python/mtlearn mtlearn/tests/python
git diff --check
```

### Notebook Validation

From a source checkout with a local CMake build:

```bash
pip install -e ".[notebooks]"
python scripts/validate_notebooks.py --bindings-dir build/mtlearn/bindings
```

The script executes the full gradcheck notebooks and creates reduced temporary
smoke copies for long experiment notebooks. Source notebooks are not modified.
By default, executed outputs are written to:

```text
/tmp/mtlearn-notebook-validation
```

To validate against an installed wheel instead of the checkout:

```bash
python scripts/validate_notebooks.py --installed-package
```

## Clean-Clone Release Checklist

Before publishing from the public repository:

```bash
git clone --recurse-submodules https://github.com/wonderalexandre/mtlearn.git
cd mtlearn
cmake -S . -B build-cpp \
      -DMTLEARN_BUILD_PYTHON=OFF \
      -DMTLEARN_WITH_TORCH=OFF \
      -DMTLEARN_BUILD_TESTS=ON
cmake --build build-cpp --parallel
ctest --test-dir build-cpp --output-on-failure
python -m build --wheel --no-isolation
python -m pip install dist/mtlearn-*.whl
python scripts/validate_notebooks.py --installed-package
```

## Release Process

Releases are built by GitHub Actions, but PyPI publication is manual.

For a production release:

1. Make sure the `CI`, `Package`, and `Notebooks` workflows are green on
   `main`.
2. Update `pyproject.toml` if the release version is changing.
3. Create and push a semantic version tag matching the package version, for
   example `v1.0.0`.
4. The `Release` workflow builds the source distribution and supported platform
   wheels, checks the package metadata, and attaches the artifacts to a GitHub
   Release.

The workflow rejects a tag when the tag version does not match
`pyproject.toml`.

The release wheel matrix currently targets Python 3.9 through 3.14 on:

- Linux x86_64;
- Windows x86_64;
- macOS arm64;
- macOS Intel x86_64.

PyPI upload is intentionally manual. Download the release artifacts from the
GitHub Release or from the workflow run, then publish them with:

```bash
python -m pip install --upgrade twine
python -m twine upload dist/*
```

Manual runs of the `Release` workflow build downloadable artifacts without
creating a GitHub Release.

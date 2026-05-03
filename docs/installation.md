# MTLearn Installation

This page covers installation paths for **MTLearn**. The published Python
package and import namespace remain `mtlearn`.

For source builds, tests, notebook validation, and releases, see
[development.md](development.md).

## Install From PyPI

```bash
pip install mtlearn
```

Verify the installation:

```bash
python - <<'PY'
import mtlearn
from mtlearn import morphology

print(mtlearn.__version__)
print(morphology.AttributeType.AREA)
PY
```

## Runtime Dependency Notes

The `mtlearn` package supports NumPy 1.x and 2.x and does not depend on
`scikit-learn` at runtime.

The native `_mtlearn` extension links against LibTorch, so the package declares
tested PyTorch ranges per Python version and platform. PyTorch no longer
publishes recent macOS Intel wheels, so that platform intentionally uses the
newest available 2.2.x line for the supported Python versions.

| Platform | Python | PyTorch requirement |
| --- | --- | --- |
| macOS Intel | 3.9 through 3.12 | `torch>=2.2.2,<2.3` |
| macOS arm64 | 3.9 | `torch>=2.8,<2.9` |
| macOS arm64 | 3.10 through 3.13 | `torch>=2.10,<2.11` |
| macOS arm64 | 3.14 | `torch>=2.11,<2.12` |
| Linux and Windows | 3.9 | `torch>=2.8,<2.9` |
| Linux and Windows | 3.10 through 3.13 | `torch>=2.10,<2.11` |
| Linux and Windows | 3.14 | `torch>=2.11,<2.12` |

Release wheels are built against the lower bound for each row and smoke-tested
against the installed LibTorch runtime before upload.

## Notebook Dependencies

Install the notebook extras when you want to run the public examples:

```bash
pip install "mtlearn[notebooks]"
```

Notebook files are not installed with the PyPI package. Clone the repository to
run the public notebooks. The main public experiment example is:

```text
notebooks/experiments/Example_screws_filtering.ipynb
```

## Source Checkout

Clone with submodules:

```bash
git clone --recurse-submodules https://github.com/wonderalexandre/MTLearn.git
cd MTLearn
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
python scripts/install_release_dependencies.py --build-tools
pip install -e .
```

For notebooks from a source checkout:

```bash
pip install -e ".[notebooks]"
```

## Next Steps

- See the root [README](../README.md) for the project overview and quick start.
- See [development.md](development.md) for source builds, tests, notebook
  validation, and releases.

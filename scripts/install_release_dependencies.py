#!/usr/bin/env python3
"""Install build/runtime dependencies used by release wheel jobs.

The release workflow intentionally builds with build isolation disabled because
the native extension must compile against the same Torch installation that will
provide the C++ headers and libraries. Keeping this logic in Python avoids
duplicating shell snippets across the native and manylinux jobs.
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys


TORCH_CPU_INDEX_URL = "https://download.pytorch.org/whl/cpu"


def run_pip(*args: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", *args])


def minimum_torch_version() -> str:
    """Return the lowest Torch version used for this Python wheel build."""

    if sys.version_info >= (3, 14):
        return "2.11.0"
    if sys.version_info >= (3, 13):
        return "2.6.0"
    if sys.version_info >= (3, 12):
        return "2.2.2"
    return "2.0.1"


def version_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def torch_requirement(mode: str) -> str | None:
    """Return the Torch requirement used by the requested install mode."""

    if mode == "none":
        return None
    if mode == "latest":
        return "torch"

    version = minimum_torch_version()
    if sys.platform.startswith("linux"):
        return f"torch=={version}+cpu"
    return f"torch=={version}"


def needs_legacy_numpy(torch_mode: str) -> bool:
    """Return whether this runner should avoid NumPy 2 during dependency setup."""

    if (
        torch_mode == "minimum"
        and version_tuple(minimum_torch_version()) < (2, 4)
    ):
        return True

    return (
        sys.version_info < (3, 13)
        and platform.system() == "Darwin"
        and platform.machine() == "x86_64"
    )


def numpy_requirement(torch_mode: str, force_legacy: bool = False) -> str:
    if force_legacy or needs_legacy_numpy(torch_mode):
        return "numpy>=1.23,<2"
    return "numpy>=1.23"


def install_build_tools() -> None:
    run_pip(
        "install",
        "build",
        "scikit-build-core",
        "setuptools-scm",
        "pybind11",
    )


def install_runtime_dependencies(torch_mode: str, force_legacy_numpy: bool) -> None:
    run_pip(
        "install",
        numpy_requirement(torch_mode, force_legacy_numpy),
        "opencv-python-headless",
        "scikit-learn",
    )


def install_torch(torch_mode: str) -> None:
    requirement = torch_requirement(torch_mode)
    if requirement is None:
        return

    args = ["install", requirement]
    if sys.platform.startswith("linux"):
        args.extend(["--index-url", TORCH_CPU_INDEX_URL])
    run_pip(*args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install mtlearn release build/runtime dependencies."
    )
    parser.add_argument(
        "--build-tools",
        action="store_true",
        help="Install Python build front-end and scikit-build dependencies.",
    )
    parser.add_argument(
        "--legacy-numpy",
        action="store_true",
        help="Install NumPy 1.x for environments with old PyTorch wheels.",
    )
    parser.add_argument(
        "--torch",
        choices=("minimum", "latest", "none"),
        default="minimum",
        help=(
            "Torch install policy. 'minimum' installs the oldest supported "
            "Torch for the active Python version, 'latest' lets pip choose the "
            "newest compatible Torch, and 'none' skips Torch installation."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_pip("install", "--upgrade", "pip")
    if args.build_tools:
        install_build_tools()
    install_runtime_dependencies(args.torch, args.legacy_numpy)
    install_torch(args.torch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

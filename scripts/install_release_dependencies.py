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


def run_pip(*args: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", *args])


def needs_legacy_numpy() -> bool:
    """Return whether this runner should avoid NumPy 2 during release builds."""
    return (
        sys.version_info < (3, 13)
        and platform.system() == "Darwin"
        and platform.machine() == "x86_64"
    )


def numpy_requirement(force_legacy: bool = False) -> str:
    if force_legacy or needs_legacy_numpy():
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


def install_runtime_dependencies(force_legacy_numpy: bool) -> None:
    run_pip(
        "install",
        numpy_requirement(force_legacy_numpy),
        "opencv-python-headless",
        "scikit-learn",
    )


def install_torch() -> None:
    args = ["install", "torch"]
    if sys.platform.startswith("linux"):
        args.extend(["--index-url", "https://download.pytorch.org/whl/cpu"])
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_pip("install", "--upgrade", "pip")
    if args.build_tools:
        install_build_tools()
    install_runtime_dependencies(args.legacy_numpy)
    install_torch()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

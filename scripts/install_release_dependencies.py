#!/usr/bin/env python3
"""Install build/runtime dependencies used by release wheel jobs.

The release workflow intentionally builds with build isolation disabled because
the native extension must compile against the same Torch installation that will
provide the C++ headers and libraries. Keeping this logic in Python avoids
duplicating shell snippets across the native and manylinux jobs.
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def run_pip(*args: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", *args])


def numpy_requirement() -> str:
    if sys.version_info < (3, 13):
        return "numpy>=1.23,<2"
    return "numpy>=2.0"


def install_build_tools() -> None:
    run_pip(
        "install",
        "build",
        "scikit-build-core",
        "setuptools-scm",
        "pybind11",
    )


def install_runtime_dependencies() -> None:
    run_pip(
        "install",
        numpy_requirement(),
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_pip("install", "--upgrade", "pip")
    if args.build_tools:
        install_build_tools()
    install_runtime_dependencies()
    install_torch()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

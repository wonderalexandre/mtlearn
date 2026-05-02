"""Native extension loading helpers."""

from __future__ import annotations

from importlib import import_module


def load_bindings():
    """Load the native bindings, preferring an in-tree CMake build when present."""
    try:
        return import_module("_mtlearn")
    except ModuleNotFoundError as top_level_error:
        if top_level_error.name != "_mtlearn":
            raise
        try:
            return import_module("._mtlearn", package=__package__)
        except ImportError as package_error:
            raise package_error from top_level_error

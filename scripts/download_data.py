#!/usr/bin/env python3
"""Command-line wrapper for mtlearn dataset downloads."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


def load_data_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "mtlearn" / "python" / "mtlearn" / "data.py"
    spec = importlib.util.spec_from_file_location("_mtlearn_data_cli", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load dataset helper from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    os.environ.setdefault("MTLEARN_DATA_DIR", str(Path(__file__).resolve().parents[1] / "dat"))
    raise SystemExit(load_data_module().main())

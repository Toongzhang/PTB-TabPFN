"""Compatibility helper for the installed TabPFN package."""

from __future__ import annotations

import importlib
from pathlib import Path


def ensure_tabpfn_importable(tabpfn_source_dir: str | Path | None = None) -> None:
    """Apply the scikit-learn naming compatibility shim used by TabPFN.

    tabpfn_source_dir is accepted for existing callers; no source path is added.
    """
    try:
        validation = importlib.import_module("sklearn.utils.validation")
        if not hasattr(validation, "_is_pandas_df") and hasattr(
            validation, "is_pandas_df"
        ):
            validation._is_pandas_df = validation.is_pandas_df
    except Exception:
        pass

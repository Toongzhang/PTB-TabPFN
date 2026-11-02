"""Final-development retraining API; no study configuration is distributed."""

from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping
from ptb_tabpfn.train import _run_training


def train_final_model(
    stage_data: Mapping[str, Any],
    *,
    settings: Mapping[str, Any],
    output_dir: str | Path,
):
    """Train the caller-selected feature set on all supplied development rows.

    Each stage must have empty validation arrays. Supply an explicit fixed
    epoch budget without validation/early stopping, and select features before
    calling this function. Returns JointFineTuningResult.
    """
    return _run_training(
        stage_data, settings=settings, output_dir=output_dir, final_training=True
    )

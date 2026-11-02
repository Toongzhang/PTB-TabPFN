"""Core joint-training API; study-specific data and settings are supplied locally."""

from __future__ import annotations
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping
import numpy as np
from ptb_tabpfn.data.inputs import validate_stage_data
from ptb_tabpfn.evaluation.metrics import binary_probability_metrics


def _run_training(
    stage_data: Mapping[str, Any],
    *,
    settings: Mapping[str, Any],
    output_dir: str | Path,
    final_training: bool,
):
    config = deepcopy(dict(settings))
    tuning = config["fine_tuning"]
    rates = tuning["learning_rates"]
    if len(rates) != 1 or not np.isfinite(rates[0]) or rates[0] <= 0:
        raise ValueError("Supply exactly one positive finite learning rate.")
    if config.get("ablation_variant", "PTB-TabPFN") != "PTB-TabPFN":
        raise ValueError("This entry trains the full PTB-TabPFN architecture.")
    config["ablation_variant"] = "PTB-TabPFN"
    config["module"] = "ptb_tabpfn"
    config["final_training"] = final_training
    epochs = tuning["epochs"]
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("Epoch budget must be a positive integer.")
    if final_training:
        if tuning["validation_epochs"] or tuning.get("early_stopping") is not None:
            raise ValueError(
                "Final training uses a fixed epoch budget without validation or early stopping."
            )
        if config.get("fixed_checkpoint_epoch") != epochs:
            raise ValueError(
                "fixed_checkpoint_epoch must equal the final epoch budget."
            )
    else:
        validation = tuning["validation_epochs"]
        if (
            not validation
            or any(
                type(epoch) is not int or not 1 <= epoch <= epochs
                for epoch in validation
            )
            or validation != sorted(set(validation))
        ):
            raise ValueError(
                "Supply an ordered, nonempty validation schedule within the epoch budget."
            )
        fixed = config.get("fixed_checkpoint_epoch")
        stopping = tuning.get("early_stopping")
        if fixed is not None:
            if fixed != epochs or fixed not in validation or stopping is not None:
                raise ValueError(
                    "A fixed checkpoint must match the epoch budget and disable early stopping."
                )
        elif not isinstance(stopping, Mapping):
            raise ValueError("Supply early stopping or a fixed checkpoint.")
        if stopping is not None and (
            stopping["metric"] != "macro_validation_pr_auc"
            or stopping["mode"] != "max"
            or not stopping["restore_best_checkpoint"]
            or stopping["patience"] < 1
            or not np.isfinite(stopping["min_delta"])
            or stopping["min_delta"] < 0
        ):
            raise ValueError(
                "Early stopping must maximize macro validation AUPRC and restore the best checkpoint."
            )
    validate_stage_data(
        stage_data,
        group_count=config["clinical_group_count"],
        require_validation=not final_training,
    )
    model_path = Path(config["model_path"]).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    config["model_path"] = str(model_path)
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)

    import torch
    from ptb_tabpfn.training.joint_training import (
        build_global_episode_plan,
        fine_tune_joint_stage_shared_private,
    )

    if str(config["device"]).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    seed = int(config.get("model_seed", config["training_seed"]))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    plan, _ = build_global_episode_plan(
        stage_data,
        max_data_size=int(tuning["max_data_size"]),
        n_folds=int(tuning["query_folds"]),
        seed=int(config["training_seed"]) + 1100,
    )
    return fine_tune_joint_stage_shared_private(
        stage_data,
        plan,
        learning_rate=float(rates[0]),
        config=config,
        output_dir=output,
        metric_callback=binary_probability_metrics,
        final_training=final_training,
    )


def train_model(
    stage_data: Mapping[str, Any],
    *,
    settings: Mapping[str, Any],
    output_dir: str | Path,
):
    """Train on prepared stage datasets using caller-supplied optimizer/validation settings.

    Returns JointFineTuningResult. This API does not load a bundled study
    configuration, select features, construct clinical groups or create CV splits.
    """
    return _run_training(
        stage_data, settings=settings, output_dir=output_dir, final_training=False
    )

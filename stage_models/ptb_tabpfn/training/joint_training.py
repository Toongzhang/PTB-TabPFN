"""Joint training of early, mid and late PTB-TabPFN inputs."""

from __future__ import annotations
import gc
import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

pd.options.future.infer_string = False
from ptb_tabpfn.training.tabpfn_runtime import (
    BinaryPrediction,
    _assert_dataset_collection_identity,
    _assert_empty_categorical_contract,
    _assert_fitted_identity_preprocessing,
    _assert_preprocessed_batch,
    _ensure_runtime_attrs,
    _forward_standard_logits,
    _inference_precision,
    _new_classifier,
    _ordered_binary_probability,
    _validate_identity_runtime_config,
    _warning_rows,
)
from ptb_tabpfn.models.stage_adaptation import (
    STAGE_NAMES,
    assert_stage_shared_private_clone_integrity,
    freeze_except_stage_shared_private,
    inject_stage_shared_private_clinical_relational,
    set_active_clinical_stage,
    stage_shared_private_parameter_audit,
    stage_shared_private_parameter_names,
)
from ptb_tabpfn.training.utils import (
    clone_state_dict_to_cpu,
    gradient_summary,
    load_cpu_state_dict,
    parameter_change_rows,
    state_change_audit,
)


@dataclass(frozen=True)
class StageDataset:
    stage: str
    X_train: np.ndarray
    y_train: np.ndarray
    train_patient_ids: np.ndarray
    X_validation: np.ndarray
    y_validation: np.ndarray
    validation_patient_ids: np.ndarray
    feature_group_ids: list[int]


@dataclass(frozen=True)
class JointPrediction:
    by_stage: dict[str, BinaryPrediction]
    metrics_by_stage: dict[str, dict[str, Any]]
    macro_metrics: dict[str, float]


@dataclass(frozen=True)
class JointFineTuningResult:
    learning_rate: float
    best_epoch: int
    early_stopping: dict[str, Any]
    prediction: JointPrediction | None
    history: pd.DataFrame
    gradient_history: pd.DataFrame
    preprocessing_audit: pd.DataFrame
    leakage_audit: pd.DataFrame
    query_coverage: pd.DataFrame
    parameter_summary: dict[str, Any]
    parameter_changes: pd.DataFrame
    warnings: list[dict[str, Any]]
    best_checkpoint: Path
    runtime_seconds: float


def build_global_episode_plan(
    stage_data: Mapping[str, StageDataset],
    *,
    max_data_size: int,
    n_folds: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Assign one episode and one query fold per unique training patient."""
    if tuple(stage_data) != STAGE_NAMES:
        raise ValueError(f"stage_data must be ordered as {STAGE_NAMES}.")
    labels_by_patient: dict[str, int] = {}
    stage_sets: dict[str, set[str]] = {}
    for stage in STAGE_NAMES:
        data = stage_data[stage]
        ids = np.asarray(data.train_patient_ids, dtype=str)
        labels = np.asarray(data.y_train, dtype=int)
        if len(ids) != len(labels) or len(set(ids.tolist())) != len(ids):
            raise ValueError(f"{stage}: invalid or duplicate training patient IDs.")
        stage_sets[stage] = set(ids.tolist())
        for patient_id, label in zip(ids, labels, strict=True):
            previous = labels_by_patient.setdefault(str(patient_id), int(label))
            if previous != int(label):
                raise RuntimeError(
                    f"Patient {patient_id} has inconsistent labels across stages."
                )
    patient_ids = np.asarray(sorted(labels_by_patient), dtype=str)
    labels = np.asarray(
        [labels_by_patient[patient_id] for patient_id in patient_ids], dtype=int
    )
    availability = np.asarray(
        [
            "".join(
                (
                    "1" if patient_id in stage_sets[stage] else "0"
                    for stage in STAGE_NAMES
                )
            )
            for patient_id in patient_ids
        ],
        dtype=str,
    )
    strata = np.asarray(
        [f"{pattern}|{label}" for pattern, label in zip(availability, labels)],
        dtype=str,
    )
    episode_count = (len(patient_ids) - 1) // int(max_data_size) + 1
    if episode_count < 1:
        raise ValueError("No training patients reached the episode planner.")
    if episode_count == 1:
        episode_indices = [np.arange(len(patient_ids), dtype=int)]
    else:
        stratum_counts = pd.Series(strata, dtype=object).value_counts()
        if int(stratum_counts.min()) < episode_count:
            raise RuntimeError(
                f"Availability/label strata cannot support the requested episode count: {stratum_counts.to_dict()}."
            )
        splitter = StratifiedKFold(
            n_splits=episode_count, shuffle=True, random_state=int(seed)
        )
        episode_indices = [
            np.asarray(test, dtype=int)
            for _, test in splitter.split(np.zeros(len(labels)), strata)
        ]
    rows: list[dict[str, Any]] = []
    for episode_index, indices in enumerate(episode_indices):
        episode_strata = strata[indices]
        counts = pd.Series(episode_strata, dtype=object).value_counts()
        if int(counts.min()) < n_folds:
            raise RuntimeError(
                f"Episode {episode_index} cannot support {n_folds} aligned query folds: {counts.to_dict()}."
            )
        fold_assignment = np.full(len(indices), -1, dtype=int)
        fold_splitter = StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=int(seed + 1000 + episode_index),
        )
        for fold, (_, query_local) in enumerate(
            fold_splitter.split(np.zeros(len(indices)), episode_strata)
        ):
            fold_assignment[query_local] = int(fold)
        if bool(np.any(fold_assignment < 0)):
            raise RuntimeError("A global patient was not assigned a query fold.")
        for local_index, global_index in enumerate(indices):
            patient_id = str(patient_ids[global_index])
            rows.append(
                {
                    "patient_id": patient_id,
                    "label": int(labels[global_index]),
                    "availability_pattern": str(availability[global_index]),
                    "episode_index": int(episode_index),
                    "query_fold": int(fold_assignment[local_index]),
                    **{
                        f"available_{stage}": bool(patient_id in stage_sets[stage])
                        for stage in STAGE_NAMES
                    },
                }
            )
    plan = pd.DataFrame(rows, dtype=object)
    for column in ("label", "episode_index", "query_fold"):
        plan[column] = plan[column].astype(np.int64)
    for stage in STAGE_NAMES:
        plan[f"available_{stage}"] = plan[f"available_{stage}"].astype(bool)
    plan = plan.sort_values(["episode_index", "patient_id"]).reset_index(drop=True)
    support_rows: list[dict[str, Any]] = []
    for episode_index in range(episode_count):
        episode = plan.loc[plan["episode_index"].eq(episode_index)]
        for fold in range(n_folds):
            global_query = set(
                episode.loc[episode["query_fold"].eq(fold), "patient_id"]
            )
            for stage in STAGE_NAMES:
                available = episode.loc[episode[f"available_{stage}"]]
                query = available.loc[available["patient_id"].isin(global_query)]
                context = available.loc[~available["patient_id"].isin(global_query)]
                query_counts = np.bincount(
                    query["label"].to_numpy(dtype=int), minlength=2
                )
                context_counts = np.bincount(
                    context["label"].to_numpy(dtype=int), minlength=2
                )
                passed = bool(
                    int(query_counts.min()) > 0 and int(context_counts.min()) > 0
                )
                support_rows.append(
                    {
                        "episode_index": int(episode_index),
                        "query_fold": int(fold),
                        "stage": stage,
                        "query_negative": int(query_counts[0]),
                        "query_positive": int(query_counts[1]),
                        "context_negative": int(context_counts[0]),
                        "context_positive": int(context_counts[1]),
                        "passed": passed,
                    }
                )
                if not passed:
                    raise RuntimeError(
                        f"Aligned episode/fold lost binary support: {support_rows[-1]}."
                    )
    sizes = plan.groupby("episode_index").size()
    audit = {
        "passed": True,
        "union_patient_count": int(len(plan)),
        "episode_count": int(episode_count),
        "minimum_episode_size": int(sizes.min()),
        "maximum_episode_size": int(sizes.max()),
        "query_fold_count": int(n_folds),
        "availability_pattern_counts": plan["availability_pattern"]
        .value_counts()
        .to_dict(),
        "support_checks": support_rows,
        "same_patient_same_query_fold_across_stages": True,
    }
    return (plan, audit)


def _stage_row_lookup(data: StageDataset) -> dict[str, int]:
    return {
        str(patient_id): int(index)
        for index, patient_id in enumerate(data.train_patient_ids)
    }


def _fixed_patient_split_fn(
    patient_ids: np.ndarray,
    fold_by_patient: Mapping[str, int],
    *,
    query_fold: int,
    global_query_ids: set[str],
    raw_audit: list[dict[str, np.ndarray]],
    patient_audit: list[dict[str, list[str]]],
):
    def split_fn(X: np.ndarray, y: np.ndarray, random_state: int | None = None):
        del random_state
        ids = np.asarray(patient_ids, dtype=str)
        folds = np.asarray(
            [int(fold_by_patient[patient_id]) for patient_id in ids], dtype=int
        )
        query_positions = np.flatnonzero(folds == int(query_fold))
        context_positions = np.flatnonzero(folds != int(query_fold))
        query_ids = set(ids[query_positions].tolist())
        context_ids = set(ids[context_positions].tolist())
        if not query_ids.issubset(global_query_ids):
            raise RuntimeError("Stage query IDs disagree with the global fold.")
        if not context_ids.isdisjoint(global_query_ids):
            raise RuntimeError("A globally queried patient entered a stage context.")
        raw_audit.append(
            {
                "context": np.asarray(X[context_positions], dtype=np.float32),
                "query": np.asarray(X[query_positions], dtype=np.float32),
            }
        )
        patient_audit.append(
            {
                "context": sorted(context_ids),
                "query": sorted(query_ids),
                "global_query": sorted(global_query_ids),
            }
        )
        return (
            X[context_positions],
            X[query_positions],
            y[context_positions],
            y[query_positions],
        )

    return split_fn


def _prepare_stage_batch(
    classifier: Any,
    data: StageDataset,
    episode: pd.DataFrame,
    *,
    query_fold: int,
    epoch: int,
    joint_step: int,
) -> tuple[tuple[Any, ...], dict[str, Any], dict[str, Any]]:
    lookup = _stage_row_lookup(data)
    stage_ids = np.asarray(
        [
            patient_id
            for patient_id in episode["patient_id"].astype(str)
            if patient_id in lookup
        ],
        dtype=str,
    )
    indices = np.asarray([lookup[patient_id] for patient_id in stage_ids])
    X_episode = np.asarray(data.X_train[indices], dtype=np.float32)
    y_episode = np.asarray(data.y_train[indices], dtype=int)
    fold_by_patient = {
        str(row.patient_id): int(row.query_fold)
        for row in episode.itertuples(index=False)
    }
    global_query_ids = set(
        episode.loc[episode["query_fold"].eq(query_fold), "patient_id"].astype(str)
    )
    raw_audit: list[dict[str, np.ndarray]] = []
    patient_audit: list[dict[str, list[str]]] = []
    datasets = classifier.get_preprocessed_datasets(
        X_episode,
        y_episode,
        _fixed_patient_split_fn(
            stage_ids,
            fold_by_patient,
            query_fold=query_fold,
            global_query_ids=global_query_ids,
            raw_audit=raw_audit,
            patient_audit=patient_audit,
        ),
        None,
        equal_split_size=True,
    )
    dataset_contract = _assert_dataset_collection_identity(
        datasets,
        X_episode,
        tag=f"ptb_tabpfn:{data.stage}:epoch={epoch}:step={joint_step}",
    )
    _assert_empty_categorical_contract(
        classifier,
        tag=f"ptb_tabpfn:{data.stage}:epoch={epoch}:step={joint_step}",
        require_fitted=True,
    )
    if len(datasets) != 1:
        raise RuntimeError("A joint stage step must contain exactly one episode.")
    from tabpfn.utils import meta_dataset_collator

    batch = meta_dataset_collator([datasets[0]])
    if len(raw_audit) != 1 or len(patient_audit) != 1:
        raise RuntimeError("Stage preprocessing audit was not executed exactly once.")
    (
        X_context_batch,
        X_query_batch,
        _y_context_batch,
        _y_query_batch,
        categorical_indices,
        ensemble_configs,
    ) = batch
    preprocessing = _assert_preprocessed_batch(
        X_context_batch,
        X_query_batch,
        categorical_indices,
        ensemble_configs,
        raw_context=raw_audit[0]["context"],
        raw_query=raw_audit[0]["query"],
        expected_feature_count=int(X_episode.shape[1]),
        epoch=epoch,
        step=joint_step,
    )
    preprocessing.update(
        {
            "stage": data.stage,
            "joint_step": int(joint_step),
            "dataset_contract_passed": bool(dataset_contract["passed"]),
        }
    )
    leakage = {
        "epoch": int(epoch),
        "joint_step": int(joint_step),
        "stage": data.stage,
        "global_query_count": int(len(patient_audit[0]["global_query"])),
        "stage_query_count": int(len(patient_audit[0]["query"])),
        "stage_context_count": int(len(patient_audit[0]["context"])),
        "stage_query_subset_of_global_query": bool(
            set(patient_audit[0]["query"]).issubset(patient_audit[0]["global_query"])
        ),
        "global_query_disjoint_from_stage_context": bool(
            set(patient_audit[0]["global_query"]).isdisjoint(
                patient_audit[0]["context"]
            )
        ),
        "passed": True,
    }
    return (batch, preprocessing, leakage)


def _macro_metrics(
    metrics_by_stage: Mapping[str, Mapping[str, Any]],
) -> dict[str, float]:
    return {
        "pr_auc": float(
            np.mean([metrics_by_stage[stage]["pr_auc"] for stage in STAGE_NAMES])
        ),
        "auc": float(
            np.mean([metrics_by_stage[stage]["auc"] for stage in STAGE_NAMES])
        ),
        "brier": float(
            np.mean([metrics_by_stage[stage]["brier"] for stage in STAGE_NAMES])
        ),
    }


def _select_joint_inference_context(
    data: StageDataset, *, config: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Select a fixed, stratified training-only context for one stage.

    This is intentionally independent of ``fine_tuning.max_data_size``.  The
    latter partitions fine-tuning episodes; this option only controls the
    context presented to TabPFN during validation/final OOF prediction.
    """
    X = np.asarray(data.X_train, dtype=np.float32)
    y = np.asarray(data.y_train, dtype=np.int64).reshape(-1)
    requested = config.get("inference_context_size")
    if requested is None or str(requested).strip().lower() == "full":
        selected = np.arange(len(y), dtype=int)
        sampling = "full_training_fold"
    else:
        try:
            size = int(requested)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "inference_context_size must be 'full', null, or an integer of at least 2."
            ) from exc
        if size < 2:
            raise ValueError("inference_context_size must be at least 2.")
        if size > len(y):
            raise ValueError(
                f"Requested inference context size {size} exceeds {data.stage} training-patient count ({len(y)})."
            )
        positive = np.flatnonzero(y == 1)
        negative = np.flatnonzero(y == 0)
        if len(positive) == 0 or len(negative) == 0:
            raise ValueError("Inference context must contain both outcome classes.")
        positive_n = int(round(size * len(positive) / len(y)))
        positive_n = min(max(positive_n, 1), size - 1)
        negative_n = size - positive_n
        if positive_n > len(positive) or negative_n > len(negative):
            raise ValueError(
                f"Requested inference context size {size} cannot preserve both classes in {data.stage}."
            )
        stage_seed = STAGE_NAMES.index(data.stage)
        seed = (
            int(config.get("inference_context_seed", config["split_seed"])) + stage_seed
        )
        rng = np.random.default_rng(seed)
        selected = np.sort(
            np.concatenate(
                [
                    rng.choice(positive, size=positive_n, replace=False),
                    rng.choice(negative, size=negative_n, replace=False),
                ]
            )
        )
        sampling = "deterministic_stratified_training_fold_subsample"
    audit = {
        "requested_context_size": "full" if requested is None else requested,
        "selected_context_size": int(len(selected)),
        "selected_positive_count": int(y[selected].sum()),
        "selected_negative_count": int(len(selected) - y[selected].sum()),
        "sampling": sampling,
        "passed": True,
    }
    return (X[selected], y[selected], audit)


def predict_joint_model(
    classifier: Any,
    stage_data: Mapping[str, StageDataset],
    *,
    config: dict[str, Any],
    metric_callback: Callable[[np.ndarray, np.ndarray], dict[str, Any]],
    tag: str,
) -> JointPrediction:
    """Evaluate one shared model separately on each native-stage context."""
    from tabpfn import TabPFNClassifier
    from tabpfn.finetune_utils import clone_model_for_evaluation

    predictions: dict[str, BinaryPrediction] = {}
    metrics: dict[str, dict[str, Any]] = {}
    for stage in STAGE_NAMES:
        data = stage_data[stage]
        X_context, y_context, context_audit = _select_joint_inference_context(
            data, config=config
        )
        set_active_clinical_stage(classifier.models_[0], stage)
        evaluation_config = {
            "device": str(config["device"]),
            "n_estimators": 1,
            "categorical_features_indices": [],
            "random_state": int(config.get("model_seed", config["split_seed"])),
            "ignore_pretraining_limits": True,
            "inference_precision": _inference_precision(config),
            "memory_saving_mode": False,
        }
        start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            evaluator = clone_model_for_evaluation(
                classifier, evaluation_config, TabPFNClassifier
            )
            evaluator.fit(X_context, y_context)
            runtime = _validate_identity_runtime_config(evaluator)
            for fitted_model in evaluator.models_:
                set_active_clinical_stage(fitted_model, stage)
            order = _assert_fitted_identity_preprocessing(
                evaluator,
                X_context,
                np.asarray(data.X_validation, dtype=np.float32),
                tag=f"{tag}:{stage}",
            )
            for fitted_model in evaluator.models_:
                for parameter in fitted_model.parameters():
                    parameter.requires_grad_(False)
            with torch.inference_mode():
                probabilities = evaluator.predict_proba(
                    np.asarray(data.X_validation, dtype=np.float32)
                )
            positive = _ordered_binary_probability(probabilities, evaluator.classes_)
        prediction = BinaryPrediction(
            probabilities=positive,
            warning_records=_warning_rows(caught, tag=f"{tag}:{stage}"),
            preprocessing_audit={
                "runtime_config": runtime,
                "feature_order": order,
                "active_stage": stage,
                "inference_context": context_audit,
                "passed": True,
            },
            runtime_seconds=float(time.perf_counter() - start),
        )
        predictions[stage] = prediction
        metrics[stage] = metric_callback(data.y_validation, positive)
        del evaluator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return JointPrediction(
        by_stage=predictions,
        metrics_by_stage=metrics,
        macro_metrics=_macro_metrics(metrics),
    )


def load_joint_checkpoint_and_predict(
    checkpoint_path: Path,
    stage_data: Mapping[str, StageDataset],
    *,
    config: dict[str, Any],
    metric_callback: Callable[[np.ndarray, np.ndarray], dict[str, Any]],
    tag: str,
) -> JointPrediction:
    """Load one locked PTB-TabPFN checkpoint for validation or final test."""
    checkpoint_path = Path(checkpoint_path).resolve()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    variant = payload.get("variant")
    if variant not in {"stage_shared_private_clinical_relational"}:
        raise ValueError(
            "Final evaluation requires a stage-shared/private clinical checkpoint."
        )
    expected_groups = {
        stage: [int(value) for value in stage_data[stage].feature_group_ids]
        for stage in STAGE_NAMES
    }
    observed_groups = {
        stage: [int(value) for value in payload["stage_feature_group_ids"][stage]]
        for stage in STAGE_NAMES
    }
    if observed_groups != expected_groups:
        raise ValueError(
            "Checkpoint stage feature-group IDs do not match the query columns."
        )
    classifier = _new_classifier(config, fit_mode="batched")
    classifier._initialize_model_variables()
    runtime = _validate_identity_runtime_config(classifier)
    if not runtime["constant_feature_preservation_enabled"]:
        raise RuntimeError("PTB-TabPFN checkpoint inference must preserve columns.")
    _assert_empty_categorical_contract(
        classifier, tag=f"{tag}:checkpoint_classifier", require_fitted=False
    )
    _ensure_runtime_attrs(classifier)
    model = classifier.models_[0]
    inject_stage_shared_private_clinical_relational(
        model,
        stage_feature_group_ids=expected_groups,
        group_count=int(config["clinical_group_count"]),
        replace_global_missing=bool(config["replace_global_missing"]),
        symmetric_relations=bool(config["symmetric_group_relations"]),
        alpha_init=float(config["alpha_init"]),
        beta_init=float(config["beta_init"]),
    )
    assert_stage_shared_private_clone_integrity(model)
    freeze_except_stage_shared_private(model)
    load_cpu_state_dict(model, payload["model_state_dict"])
    model.eval()
    prediction = predict_joint_model(
        classifier, stage_data, config=config, metric_callback=metric_callback, tag=tag
    )
    del classifier, model, payload
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return prediction


def fine_tune_joint_stage_shared_private(
    stage_data: Mapping[str, StageDataset],
    episode_plan: pd.DataFrame,
    *,
    learning_rate: float,
    config: dict[str, Any],
    output_dir: Path,
    metric_callback: Callable[[np.ndarray, np.ndarray], dict[str, Any]],
    final_training: bool = False,
) -> JointFineTuningResult:
    """Run stage-native forwards and one optimizer update per joint step."""
    output_dir.mkdir(parents=True, exist_ok=False)
    tuning = config["fine_tuning"]
    classifier = _new_classifier(config, fit_mode="batched")
    classifier._initialize_model_variables()
    runtime_audit = _validate_identity_runtime_config(classifier)
    if not runtime_audit["constant_feature_preservation_enabled"]:
        raise RuntimeError(
            "PTB-TabPFN must preserve context-constant semantic features."
        )
    declared_audit = _assert_empty_categorical_contract(
        classifier, tag="ptb_tabpfn_joint_classifier", require_fitted=False
    )
    _ensure_runtime_attrs(classifier)
    model = classifier.models_[0]
    stage_groups = {stage: stage_data[stage].feature_group_ids for stage in STAGE_NAMES}
    injection = inject_stage_shared_private_clinical_relational(
        model,
        stage_feature_group_ids=stage_groups,
        group_count=int(config["clinical_group_count"]),
        replace_global_missing=bool(config["replace_global_missing"]),
        symmetric_relations=bool(config["symmetric_group_relations"]),
        alpha_init=float(config["alpha_init"]),
        beta_init=float(config["beta_init"]),
    )
    assert_stage_shared_private_clone_integrity(model)
    parameter_summary = freeze_except_stage_shared_private(model)
    if not parameter_summary["only_stage_shared_private_parameters_trainable"]:
        raise RuntimeError("Unexpected pretrained parameter remains trainable.")
    ablation_variant = str(config.get("ablation_variant", "PTB-TabPFN"))
    if ablation_variant not in {"M2", "PTB-TabPFN"}:
        raise ValueError(f"Unsupported joint ablation variant: {ablation_variant}")
    frozen_patterns: tuple[str, ...]
    if ablation_variant == "M2":
        frozen_patterns = (
            "stage_missing_residual.",
            "stage_group_relation_delta.",
            "clinical_relation_beta.",
        )
    else:
        frozen_patterns = ()
    for name, parameter in model.named_parameters():
        if any((pattern in name for pattern in frozen_patterns)):
            parameter.requires_grad_(False)
    trainable_after_ablation = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    all_new_parameters = stage_shared_private_parameter_names(model)
    if not trainable_after_ablation:
        raise RuntimeError(f"{ablation_variant} has no trainable parameters.")
    if not trainable_after_ablation.issubset(all_new_parameters):
        raise RuntimeError(f"{ablation_variant} left a pretrained parameter trainable.")
    parameter_summary["ablation_variant"] = ablation_variant
    parameter_summary["ablation_frozen_patterns"] = list(frozen_patterns)
    parameter_summary["ablation_trainable_parameter_names"] = sorted(
        trainable_after_ablation
    )
    parameter_summary["ablation_trainable_parameter_count"] = int(
        sum(
            (
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            )
        )
    )
    parameter_summary.update(
        {
            "injection": injection.as_dict(),
            "runtime_preprocessing": runtime_audit,
            "declared_categorical_contract": declared_audit,
            "joint_loss": "(L_early + L_mid + L_late) / 3",
            "backward_implementation": "three sequential loss/3 backward calls, one optimizer step; mathematically identical to one summed backward and lower memory",
        }
    )
    allowed_parameters = stage_shared_private_parameter_names(model)
    initial_state = clone_state_dict_to_cpu(model)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(learning_rate),
        weight_decay=float(tuning["weight_decay"]),
    )
    loss_function = torch.nn.BCEWithLogitsLoss()
    epochs = int(tuning["epochs"])
    folds = int(tuning["query_folds"])
    validation_epochs = {int(value) for value in tuning["validation_epochs"]}
    fixed_checkpoint_epoch = config.get("fixed_checkpoint_epoch")
    if final_training:
        if validation_epochs:
            raise ValueError(
                "Final development training must not evaluate a validation set."
            )
        if tuning.get("early_stopping") is not None:
            raise ValueError(
                "Final development training must not use validation early stopping."
            )
        if fixed_checkpoint_epoch is None or int(fixed_checkpoint_epoch) != epochs:
            raise ValueError(
                "Final development training requires fixed_checkpoint_epoch to equal the fixed epoch budget."
            )
        fixed_checkpoint_epoch = int(fixed_checkpoint_epoch)
    elif fixed_checkpoint_epoch is not None:
        fixed_checkpoint_epoch = int(fixed_checkpoint_epoch)
        if fixed_checkpoint_epoch not in validation_epochs:
            raise ValueError(
                "fixed_checkpoint_epoch must be present in validation_epochs."
            )
    early_stopping_config = tuning.get("early_stopping")
    early_stopping_enabled = early_stopping_config is not None
    if early_stopping_enabled:
        early_stopping_config = dict(early_stopping_config)
        early_stopping_patience = int(early_stopping_config["patience"])
        early_stopping_min_delta = float(early_stopping_config["min_delta"])
    else:
        early_stopping_patience = 0
        early_stopping_min_delta = 0.0
    history_rows: list[dict[str, Any]] = []
    gradient_rows: list[dict[str, Any]] = []
    preprocessing_rows: list[dict[str, Any]] = []
    leakage_rows: list[dict[str, Any]] = []
    warnings_all: list[dict[str, Any]] = []
    coverage: dict[tuple[str, str], list[int]] = {
        (stage, str(patient_id)): []
        for stage in STAGE_NAMES
        for patient_id in stage_data[stage].train_patient_ids
    }
    best_score: tuple[float, float, float] | None = None
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    best_monitor_value: float | None = None
    best_monitor_epoch = -1
    consecutive_non_improvements = 0
    stopped_early = False
    stop_epoch: int | None = None
    completed_epochs = 0
    start = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    episode_indices = sorted(episode_plan["episode_index"].unique().tolist())
    for epoch in range(1, epochs + 1):
        completed_epochs = epoch
        early_stopping_triggered = False
        query_fold = int((epoch - 1) % folds)
        epoch_losses = {stage: [] for stage in STAGE_NAMES}
        for joint_step, episode_index in enumerate(episode_indices, start=1):
            episode = episode_plan.loc[
                episode_plan["episode_index"].eq(episode_index)
            ].copy()
            global_query_ids = set(
                episode.loc[episode["query_fold"].eq(query_fold), "patient_id"].astype(
                    str
                )
            )
            optimizer.zero_grad(set_to_none=True)
            stage_query_sets: dict[str, set[str]] = {}
            for stage in STAGE_NAMES:
                data = stage_data[stage]
                batch, preprocessing, leakage = _prepare_stage_batch(
                    classifier,
                    data,
                    episode,
                    query_fold=query_fold,
                    epoch=epoch,
                    joint_step=joint_step,
                )
                preprocessing_rows.append(preprocessing)
                leakage_rows.append(leakage)
                (
                    X_context_batch,
                    X_query_batch,
                    y_context_batch,
                    y_query_batch,
                    categorical_indices,
                    ensemble_configs,
                ) = batch
                query_labels = (
                    y_query_batch.detach().cpu().numpy().reshape(-1).astype(int)
                )
                context_labels = (
                    y_context_batch[0].detach().cpu().numpy().reshape(-1).astype(int)
                )
                if set(np.unique(query_labels).tolist()) != {0, 1}:
                    raise RuntimeError(f"{stage}: query lost a binary class.")
                if set(np.unique(context_labels).tolist()) != {0, 1}:
                    raise RuntimeError(f"{stage}: context lost a binary class.")
                stage_available = set(
                    np.asarray(data.train_patient_ids, dtype=str).tolist()
                )
                stage_query = global_query_ids & stage_available
                stage_query_sets[stage] = stage_query
                for patient_id in stage_query:
                    coverage[stage, patient_id].append(epoch)
                set_active_clinical_stage(model, stage)
                classifier.fit_from_preprocessed(
                    X_context_batch,
                    y_context_batch,
                    categorical_indices,
                    ensemble_configs,
                )
                _ensure_runtime_attrs(classifier)
                logits = _forward_standard_logits(classifier, X_query_batch)
                binary_logits = (logits[:, 1] - logits[:, 0]) / float(
                    classifier.softmax_temperature_
                )
                targets = torch.as_tensor(
                    query_labels, device=binary_logits.device, dtype=binary_logits.dtype
                )
                stage_loss = loss_function(binary_logits, targets)
                if not bool(torch.isfinite(stage_loss).item()):
                    raise RuntimeError(f"{stage}: non-finite joint loss.")
                (stage_loss / float(len(STAGE_NAMES))).backward()
                epoch_losses[stage].append(float(stage_loss.detach().cpu().item()))
            for stage in STAGE_NAMES:
                if not stage_query_sets[stage].issubset(global_query_ids):
                    raise RuntimeError("Stage query routing violated global fold.")
            gradients = gradient_summary(model)
            if not gradients["all_observed_gradients_finite"]:
                raise RuntimeError("PTB-TabPFN produced non-finite gradients.")
            clipped = torch.nn.utils.clip_grad_norm_(
                trainable_parameters, float(tuning["gradient_clip_norm"])
            )
            optimizer.step()
            gradient_rows.append(
                {
                    "learning_rate": float(learning_rate),
                    "adaptation_learning_rate": float(learning_rate),
                    "epoch": int(epoch),
                    "query_fold": int(query_fold + 1),
                    "joint_step": int(joint_step),
                    "loss_early": epoch_losses["early"][-1],
                    "loss_mid": epoch_losses["mid"][-1],
                    "loss_late": epoch_losses["late"][-1],
                    "joint_loss": float(
                        np.mean([epoch_losses[stage][-1] for stage in STAGE_NAMES])
                    ),
                    "preclip_gradient_l2_norm": float(gradients["gradient_l2_norm"]),
                    "clip_grad_returned_norm": float(clipped.detach().cpu().item()),
                    "parameter_tensors_with_nonzero_grad": int(
                        gradients["parameter_tensors_with_nonzero_grad"]
                    ),
                    "cuda_max_allocated_mib": float(
                        torch.cuda.max_memory_allocated() / 1024**2
                    )
                    if torch.cuda.is_available()
                    else 0.0,
                }
            )
        row: dict[str, Any] = {
            "learning_rate": float(learning_rate),
            "epoch": int(epoch),
            "query_fold": int(query_fold + 1),
            "joint_step_count": int(len(episode_indices)),
            **{
                f"mean_loss_{stage}": float(np.mean(epoch_losses[stage]))
                for stage in STAGE_NAMES
            },
            "mean_joint_loss": float(
                np.mean(
                    [value for stage in STAGE_NAMES for value in epoch_losses[stage]]
                )
            ),
            "validation_evaluated": bool(epoch in validation_epochs),
        }
        if final_training and epoch == epochs:
            best_epoch = epoch
            best_state = clone_state_dict_to_cpu(model)
        elif epoch in validation_epochs:
            model.eval()
            prediction = predict_joint_model(
                classifier,
                stage_data,
                config=config,
                metric_callback=metric_callback,
                tag=f"lr={learning_rate}:epoch={epoch}:validation",
            )
            for stage in STAGE_NAMES:
                warnings_all.extend(prediction.by_stage[stage].warning_records)
                for metric_name, value in prediction.metrics_by_stage[stage].items():
                    row[f"validation_{stage}_{metric_name}"] = value
            for metric_name, value in prediction.macro_metrics.items():
                row[f"validation_macro_{metric_name}"] = value
            score = (
                float(prediction.macro_metrics["pr_auc"]),
                -float(prediction.macro_metrics["brier"]),
                float(prediction.macro_metrics["auc"]),
            )
            if fixed_checkpoint_epoch is not None and epoch == fixed_checkpoint_epoch:
                best_score = score
                best_epoch = epoch
                best_state = clone_state_dict_to_cpu(model)
            elif fixed_checkpoint_epoch is None and (
                best_score is None or score > best_score
            ):
                best_score = score
                best_epoch = epoch
                best_state = clone_state_dict_to_cpu(model)
            if early_stopping_enabled:
                monitor_value = float(prediction.macro_metrics["pr_auc"])
                monitor_improved = (
                    best_monitor_value is None
                    or monitor_value > best_monitor_value + early_stopping_min_delta
                )
                if monitor_improved:
                    best_monitor_value = monitor_value
                    best_monitor_epoch = epoch
                    consecutive_non_improvements = 0
                else:
                    consecutive_non_improvements += 1
                early_stopping_triggered = (
                    consecutive_non_improvements >= early_stopping_patience
                )
                row.update(
                    {
                        "early_stopping_monitor": "macro_validation_pr_auc",
                        "early_stopping_monitor_value": monitor_value,
                        "early_stopping_min_delta": early_stopping_min_delta,
                        "early_stopping_patience": early_stopping_patience,
                        "early_stopping_improved": bool(monitor_improved),
                        "early_stopping_consecutive_non_improvements": int(
                            consecutive_non_improvements
                        ),
                        "early_stopping_triggered": bool(early_stopping_triggered),
                    }
                )
            model.train()
        history_rows.append(row)
        if early_stopping_triggered:
            stopped_early = True
            stop_epoch = epoch
            break
    coverage_rows = [
        {
            "stage": stage,
            "patient_id": patient_id,
            "query_count": int(len(epoch_values)),
            "query_epochs": ";".join((str(value) for value in epoch_values)),
        }
        for (stage, patient_id), epoch_values in coverage.items()
    ]
    coverage_frame = pd.DataFrame(coverage_rows)
    minimum_required = int(tuning["minimum_query_count_per_patient"])
    if int(coverage_frame["query_count"].min()) < minimum_required:
        raise RuntimeError("PTB-TabPFN query coverage gate failed.")
    if best_state is None or best_epoch < 0:
        raise RuntimeError(
            "PTB-TabPFN produced no final checkpoint at the prescribed epoch."
            if final_training
            else "PTB-TabPFN produced no validation checkpoint."
        )
    early_stopping = {
        "enabled": bool(early_stopping_enabled),
        "metric": "macro_validation_pr_auc" if early_stopping_enabled else None,
        "mode": "max" if early_stopping_enabled else None,
        "patience": int(early_stopping_patience) if early_stopping_enabled else None,
        "min_delta": float(early_stopping_min_delta)
        if early_stopping_enabled
        else None,
        "restore_best_checkpoint": bool(early_stopping_enabled),
        "best_monitor_epoch": int(best_monitor_epoch)
        if early_stopping_enabled
        else None,
        "best_monitor_value": float(best_monitor_value)
        if best_monitor_value is not None
        else None,
        "stopped_early": bool(stopped_early),
        "stop_epoch": int(stop_epoch) if stop_epoch is not None else None,
        "completed_epochs": int(completed_epochs),
        "stop_reason": "patience_exhausted"
        if stopped_early
        else "max_epochs_reached"
        if early_stopping_enabled
        else "disabled",
        "fixed_checkpoint_epoch": fixed_checkpoint_epoch,
        "checkpoint_selection": "final_prespecified_epoch"
        if final_training
        else "prespecified_epoch"
        if fixed_checkpoint_epoch is not None
        else "validation_metric",
    }
    load_cpu_state_dict(model, best_state)
    model.eval()
    final_prediction: JointPrediction | None = None
    if not final_training:
        final_prediction = predict_joint_model(
            classifier,
            stage_data,
            config=config,
            metric_callback=metric_callback,
            tag=f"lr={learning_rate}:best_epoch={best_epoch}:validation",
        )
        for stage in STAGE_NAMES:
            warnings_all.extend(final_prediction.by_stage[stage].warning_records)
    final_state = clone_state_dict_to_cpu(model)
    state_audit = state_change_audit(
        initial_state, final_state, allowed=allowed_parameters
    )
    if not state_audit["only_allowed_parameters_changed"]:
        raise RuntimeError(f"Unexpected model state change: {state_audit}.")
    parameter_summary.update(
        {
            "state_change_audit": state_audit,
            "final_parameter_audit": stage_shared_private_parameter_audit(model),
            "preprocessing_episode_count": int(len(preprocessing_rows)),
            "all_preprocessing_audits_passed": bool(
                all((row["passed"] for row in preprocessing_rows))
            ),
            "all_leakage_audits_passed": bool(
                all((row["passed"] for row in leakage_rows))
            ),
            "final_validation_preprocessing": None
            if final_training
            else {
                stage: final_prediction.by_stage[stage].preprocessing_audit
                for stage in STAGE_NAMES
            },
            "final_training_without_validation": bool(final_training),
            "early_stopping": early_stopping,
        }
    )
    checkpoint = output_dir / "best_checkpoint.pt"
    torch.save(
        {
            "module": config["module"],
            "variant": "stage_shared_private_clinical_relational",
            "learning_rate": float(learning_rate),
            "adaptation_learning_rate": float(learning_rate),
            "best_epoch": int(best_epoch),
            "fixed_checkpoint_epoch": fixed_checkpoint_epoch,
            "checkpoint_selection": "final_prespecified_epoch"
            if final_training
            else "prespecified_epoch"
            if fixed_checkpoint_epoch is not None
            else "validation_metric",
            "early_stopping": early_stopping,
            "stage_feature_group_ids": {
                stage: stage_data[stage].feature_group_ids for stage in STAGE_NAMES
            },
            "model_state_dict": best_state,
            "config": config,
            "test_predictions_computed": False,
        },
        checkpoint,
    )
    history = pd.DataFrame(history_rows)
    gradients = pd.DataFrame(gradient_rows)
    preprocessing_frame = pd.DataFrame(preprocessing_rows)
    leakage_frame = pd.DataFrame(leakage_rows)
    changes = pd.DataFrame(parameter_change_rows(initial_state, final_state))
    history.to_csv(output_dir / "training_history.csv", index=False)
    gradients.to_csv(output_dir / "gradient_norm_history.csv", index=False)
    preprocessing_frame.to_json(
        output_dir / "preprocessing_episode_audit.jsonl", orient="records", lines=True
    )
    leakage_frame.to_csv(output_dir / "cross_stage_leakage_audit.csv", index=False)
    coverage_frame.to_csv(output_dir / "query_coverage.csv", index=False)
    changes.to_csv(output_dir / "parameter_change_by_layer.csv", index=False)
    (output_dir / "parameter_summary.json").write_text(
        json.dumps(parameter_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "warnings.json").write_text(
        json.dumps({"warnings": warnings_all}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    result = JointFineTuningResult(
        learning_rate=float(learning_rate),
        best_epoch=int(best_epoch),
        early_stopping=early_stopping,
        prediction=final_prediction,
        history=history,
        gradient_history=gradients,
        preprocessing_audit=preprocessing_frame,
        leakage_audit=leakage_frame,
        query_coverage=coverage_frame,
        parameter_summary=parameter_summary,
        parameter_changes=changes,
        warnings=warnings_all,
        best_checkpoint=checkpoint.resolve(),
        runtime_seconds=float(time.perf_counter() - start),
    )
    del classifier, model, optimizer, best_state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


__all__ = [
    "JointFineTuningResult",
    "JointPrediction",
    "StageDataset",
    "build_global_episode_plan",
    "fine_tune_joint_stage_shared_private",
    "load_joint_checkpoint_and_predict",
    "predict_joint_model",
]

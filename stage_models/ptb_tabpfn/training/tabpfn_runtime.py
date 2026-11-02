"""TabPFN preprocessing checks and forward helpers used by joint training."""

from __future__ import annotations
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import torch
from ptb_tabpfn.models.tabpfn_backend import ensure_tabpfn_importable


@dataclass(frozen=True)
class BinaryPrediction:
    probabilities: np.ndarray
    warning_records: list[dict[str, Any]]
    preprocessing_audit: dict[str, Any]
    runtime_seconds: float


def _inference_precision(config: dict[str, Any]) -> torch.dtype:
    value = str(config.get("inference_precision", "float32")).lower()
    if value != "float32":
        raise ValueError(
            "clinical adaptation is locked to float32 for its first experiment."
        )
    return torch.float32


def _identity_inference_config() -> dict[str, Any]:
    from tabpfn.preprocessing import PreprocessorConfig

    return {
        "CLASS_SHIFT_METHOD": None,
        "FEATURE_SHIFT_METHOD": None,
        "FINGERPRINT_FEATURE": False,
        "MIN_UNIQUE_FOR_NUMERICAL_FEATURES": 0,
        "POLYNOMIAL_FEATURES": "no",
        "PREPROCESS_TRANSFORMS": [
            PreprocessorConfig(
                name="none",
                append_original=False,
                categorical_name="numeric",
                global_transformer_name=None,
                max_features_per_estimator=500,
            )
        ],
    }


def _enable_constant_feature_preservation() -> None:
    """Keep semantic columns even when one episode context is constant.

    TabPFN normally removes columns that are constant in the current context.
    Clinical group IDs are position-aligned, so PTB-TabPFN must retain those
    columns. This runtime patch is opt-in through the preprocessing contract.
    """
    from tabpfn.preprocessors.remove_constant_features_step import (
        RemoveConstantFeaturesStep,
    )

    if bool(
        getattr(RemoveConstantFeaturesStep, "_ptb_tabpfn_preserve_all_features", False)
    ):
        return
    original_fit = RemoveConstantFeaturesStep._fit

    def _fit_keep_all(
        self: Any, X: np.ndarray | torch.Tensor, categorical_features: list[int]
    ) -> list[int]:
        self.sel_ = [True] * int(X.shape[1])
        return [int(value) for value in categorical_features]

    RemoveConstantFeaturesStep._ptb_tabpfn_original_fit = original_fit
    RemoveConstantFeaturesStep._fit = _fit_keep_all
    RemoveConstantFeaturesStep._ptb_tabpfn_preserve_all_features = True


def _new_classifier(config: dict[str, Any], *, fit_mode: str):
    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
    ensure_tabpfn_importable(config.get("tabpfn_source_dir"))
    if bool(
        config.get("preprocessing_contract", {}).get(
            "preserve_constant_features", False
        )
    ):
        _enable_constant_feature_preservation()
    from tabpfn import TabPFNClassifier

    return TabPFNClassifier(
        device=str(config["device"]),
        model_path=str(Path(config["model_path"]).resolve()),
        n_estimators=1,
        categorical_features_indices=[],
        random_state=int(config.get("model_seed", config["split_seed"])),
        fit_mode=fit_mode,
        differentiable_input=False,
        ignore_pretraining_limits=True,
        inference_precision=_inference_precision(config),
        memory_saving_mode=False,
        inference_config=_identity_inference_config(),
    )


def _validate_identity_runtime_config(classifier: Any) -> dict[str, Any]:
    from tabpfn.preprocessors.remove_constant_features_step import (
        RemoveConstantFeaturesStep,
    )

    runtime = classifier.inference_config_
    preprocessors = list(runtime.PREPROCESS_TRANSFORMS)
    rows = [
        {
            "name": str(item.name),
            "append_original": item.append_original,
            "categorical_name": str(item.categorical_name),
            "global_transformer_name": item.global_transformer_name,
            "max_features_per_estimator": int(item.max_features_per_estimator),
        }
        for item in preprocessors
    ]
    passed = bool(
        runtime.CLASS_SHIFT_METHOD is None
        and runtime.FEATURE_SHIFT_METHOD is None
        and (runtime.FINGERPRINT_FEATURE is False)
        and (runtime.MIN_UNIQUE_FOR_NUMERICAL_FEATURES == 0)
        and (runtime.POLYNOMIAL_FEATURES == "no")
        and (len(preprocessors) == 1)
        and (preprocessors[0].name == "none")
        and (preprocessors[0].append_original is False)
        and (preprocessors[0].categorical_name == "numeric")
        and (preprocessors[0].global_transformer_name is None)
    )
    audit = {
        "passed": passed,
        "class_shift_method": runtime.CLASS_SHIFT_METHOD,
        "feature_shift_method": runtime.FEATURE_SHIFT_METHOD,
        "fingerprint_feature": bool(runtime.FINGERPRINT_FEATURE),
        "min_unique_for_numerical_features": int(
            runtime.MIN_UNIQUE_FOR_NUMERICAL_FEATURES
        ),
        "polynomial_features": runtime.POLYNOMIAL_FEATURES,
        "preprocessors": rows,
        "constant_feature_preservation_enabled": bool(
            getattr(
                RemoveConstantFeaturesStep, "_ptb_tabpfn_preserve_all_features", False
            )
        ),
    }
    if not passed:
        raise RuntimeError(f"Semantic-preserving preprocessing lock failed: {audit}.")
    return audit


def _assert_empty_categorical_contract(
    classifier: Any, *, tag: str, require_fitted: bool
) -> dict[str, Any]:
    declared = getattr(classifier, "categorical_features_indices", None)
    declared_list = None if declared is None else [int(value) for value in declared]
    if declared_list != []:
        raise RuntimeError(
            f"{tag}: categorical_features_indices must be explicitly [], got {declared_list!r}."
        )
    inferred: list[int] | None = None
    if require_fitted:
        if not hasattr(classifier, "inferred_categorical_indices_"):
            raise RuntimeError(
                f"{tag}: fitted classifier has no inferred_categorical_indices_."
            )
        inferred = [int(value) for value in classifier.inferred_categorical_indices_]
        if inferred:
            raise RuntimeError(
                f"{tag}: TabPFN inferred categorical columns despite the all-numeric contract: {inferred}."
            )
    return {
        "tag": tag,
        "declared_categorical_features_indices": declared_list,
        "inferred_categorical_indices": inferred,
        "passed": True,
    }


def _as_feature_matrix(value: np.ndarray | torch.Tensor, *, tag: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if array.ndim == 3 and int(array.shape[0]) == 1:
        array = array[0]
    if array.ndim != 2:
        raise RuntimeError(
            f"{tag}: expected a 2D feature matrix (or leading singleton batch), got shape {tuple(array.shape)}."
        )
    return np.asarray(array)


def _assert_exact_feature_matrix(
    observed: np.ndarray | torch.Tensor,
    expected: np.ndarray | torch.Tensor,
    *,
    tag: str,
) -> dict[str, Any]:
    observed_array = _as_feature_matrix(observed, tag=f"{tag}:observed")
    expected_array = _as_feature_matrix(expected, tag=f"{tag}:expected")
    if observed_array.shape != expected_array.shape:
        raise RuntimeError(
            f"{tag}: preprocessing changed matrix shape from {tuple(expected_array.shape)} to {tuple(observed_array.shape)}."
        )
    matches = np.equal(observed_array, expected_array)
    try:
        matches |= np.isnan(observed_array) & np.isnan(expected_array)
    except TypeError:
        pass
    column_matches = np.all(matches, axis=0)
    if not bool(np.all(column_matches)):
        mismatched = np.flatnonzero(~column_matches).astype(int).tolist()
        raise RuntimeError(
            f"{tag}: preprocessing changed values or column order at {len(mismatched)} positions; first mismatches={mismatched[:12]}."
        )
    return {
        "tag": tag,
        "row_count": int(expected_array.shape[0]),
        "feature_count": int(expected_array.shape[1]),
        "mismatched_feature_positions": [],
        "exact_value_and_order_match": True,
        "passed": True,
    }


def _assert_preprocessor_lineage(
    preprocessor: Any, *, expected_feature_count: int, tag: str
) -> dict[str, Any]:
    expected = np.arange(int(expected_feature_count), dtype=int)
    rows: list[dict[str, Any]] = []
    for step_index, step in enumerate(preprocessor.steps):
        row: dict[str, Any] = {
            "step_index": int(step_index),
            "step_type": type(step).__name__,
        }
        if hasattr(step, "sel_"):
            selection = np.asarray(step.sel_, dtype=bool)
            if selection.shape != (expected_feature_count,) or not bool(
                np.all(selection)
            ):
                raise RuntimeError(f"{tag}: {type(step).__name__} removed a feature.")
            row["selected_feature_indices"] = expected.tolist()
        if hasattr(step, "subsampled_features_"):
            selected = np.asarray(step.subsampled_features_, dtype=int)
            if not np.array_equal(selected, expected):
                raise RuntimeError(
                    f"{tag}: {type(step).__name__} changed feature lineage to {selected.tolist()}."
                )
            row["subsampled_feature_indices"] = selected.tolist()
        if hasattr(step, "index_permutation_"):
            permutation = np.asarray(step.index_permutation_, dtype=int)
            if not np.array_equal(permutation, expected):
                raise RuntimeError(
                    f"{tag}: {type(step).__name__} changed feature order to {permutation.tolist()}."
                )
            row["feature_permutation"] = permutation.tolist()
        rows.append(row)
    return {
        "tag": tag,
        "feature_count": int(expected_feature_count),
        "steps": rows,
        "identity_lineage": True,
        "passed": True,
    }


def _assert_fitted_identity_preprocessing(
    classifier: Any, X_train: np.ndarray, X_query: np.ndarray, *, tag: str
) -> dict[str, Any]:
    categorical = _assert_empty_categorical_contract(
        classifier, tag=tag, require_fitted=True
    )
    engine = classifier.executor_
    if (
        len(engine.X_trains) != 1
        or len(engine.preprocessors) != 1
        or len(engine.cat_ixs) != 1
    ):
        raise RuntimeError(
            f"{tag}: clinical adaptation requires exactly one preprocessor."
        )
    post_transform_cat_ix = [int(value) for value in engine.cat_ixs[0]]
    if post_transform_cat_ix:
        raise RuntimeError(
            f"{tag}: preprocessor returned categorical indices {post_transform_cat_ix}."
        )
    train_matrix = _assert_exact_feature_matrix(
        engine.X_trains[0], X_train, tag=f"{tag}:train"
    )
    query_transformed = (
        engine.preprocessors[0].transform(np.asarray(X_query, dtype=np.float32)).X
    )
    query_matrix = _assert_exact_feature_matrix(
        query_transformed, X_query, tag=f"{tag}:query"
    )
    lineage = _assert_preprocessor_lineage(
        engine.preprocessors[0], expected_feature_count=int(X_train.shape[1]), tag=tag
    )
    return {
        "tag": tag,
        "categorical_contract": categorical,
        "post_transform_categorical_indices": post_transform_cat_ix,
        "train_matrix": train_matrix,
        "query_matrix": query_matrix,
        "preprocessor_lineage": lineage,
        "passed": True,
    }


def _warning_rows(
    caught: list[warnings.WarningMessage], *, tag: str
) -> list[dict[str, Any]]:
    return [
        {
            "tag": tag,
            "category": item.category.__name__,
            "message": str(item.message),
            "filename": str(item.filename),
            "lineno": int(item.lineno),
        }
        for item in caught
    ]


def _ordered_binary_probability(
    probabilities: np.ndarray, classes: np.ndarray | list[Any]
) -> np.ndarray:
    matrix = np.asarray(probabilities, dtype=float)
    observed = [int(value) for value in np.asarray(classes).tolist()]
    if set(observed) != {0, 1}:
        raise RuntimeError(f"Expected binary classes [0, 1], got {observed}.")
    positive = matrix[:, observed.index(1)]
    if not np.isfinite(positive).all():
        raise RuntimeError("TabPFN returned non-finite probabilities.")
    return positive


def _ensure_runtime_attrs(classifier: Any) -> None:
    if not hasattr(classifier, "softmax_temperature_"):
        classifier.softmax_temperature_ = classifier.softmax_temperature
    if not hasattr(classifier, "tuned_classification_thresholds_"):
        classifier.tuned_classification_thresholds_ = None


def _assert_dataset_collection_identity(
    datasets: Any, X_ordered: np.ndarray, *, tag: str
) -> dict[str, Any]:
    raw_parts: list[np.ndarray] = []
    categorical_by_dataset: list[list[int]] = []
    for dataset_index, dataset_config in enumerate(datasets.configs):
        categorical = [int(value) for value in dataset_config.cat_ix]
        if categorical:
            raise RuntimeError(
                f"{tag}: dataset {dataset_index} inferred categorical indices {categorical}."
            )
        categorical_by_dataset.append(categorical)
        raw_parts.append(
            _as_feature_matrix(
                dataset_config.X_raw, tag=f"{tag}:dataset={dataset_index}"
            )
        )
    if not raw_parts:
        raise RuntimeError(f"{tag}: preprocessing produced no episode datasets.")
    reconstructed = np.concatenate(raw_parts, axis=0)
    matrix = _assert_exact_feature_matrix(
        reconstructed, X_ordered, tag=f"{tag}:pre_dataset_order"
    )
    return {
        "tag": tag,
        "dataset_count": int(len(raw_parts)),
        "categorical_indices_by_dataset": categorical_by_dataset,
        "pre_dataset_matrix": matrix,
        "passed": True,
    }


def _flatten_integer_indices(value: Any) -> list[int]:
    if isinstance(value, (list, tuple)):
        flattened: list[int] = []
        for item in value:
            flattened.extend(_flatten_integer_indices(item))
        return flattened
    if isinstance(value, np.ndarray):
        return [int(item) for item in value.reshape(-1).tolist()]
    if isinstance(value, (int, np.integer)):
        return [int(value)]
    if value is None:
        return []
    raise RuntimeError(f"Unexpected categorical-index payload {type(value).__name__}.")


def _assert_preprocessed_batch(
    X_context_batch: list[torch.Tensor],
    X_query_batch: list[torch.Tensor],
    categorical_indices: Any,
    ensemble_configs: list[list[Any]],
    *,
    raw_context: np.ndarray,
    raw_query: np.ndarray,
    expected_feature_count: int,
    epoch: int,
    step: int,
) -> dict[str, Any]:
    if len(X_context_batch) != 1 or len(X_query_batch) != 1:
        raise RuntimeError("clinical adaptation requires batch_size=n_estimators=1.")
    observed_context = int(X_context_batch[0].shape[-1])
    observed_query = int(X_query_batch[0].shape[-1])
    if (
        observed_context != expected_feature_count
        or observed_query != expected_feature_count
    ):
        raise RuntimeError(
            f"TabPFN preprocessing changed the feature count; clinical group IDs would be misaligned. raw={expected_feature_count}, context={observed_context}, query={observed_query}, epoch={epoch}, step={step}."
        )
    flattened_categorical = _flatten_integer_indices(categorical_indices)
    if flattened_categorical:
        raise RuntimeError(
            f"An episode retained categorical indices after preprocessing: {flattened_categorical}, epoch={epoch}, step={step}."
        )
    context_identity = _assert_exact_feature_matrix(
        X_context_batch[0],
        raw_context,
        tag=f"training:epoch={epoch}:step={step}:context",
    )
    query_identity = _assert_exact_feature_matrix(
        X_query_batch[0], raw_query, tag=f"training:epoch={epoch}:step={step}:query"
    )
    config_rows: list[dict[str, Any]] = []
    for configs in ensemble_configs:
        for ensemble_config in configs:
            row = {
                "class_permutation": None
                if ensemble_config.class_permutation is None
                else np.asarray(ensemble_config.class_permutation).tolist(),
                "feature_shift_decoder": ensemble_config.feature_shift_decoder,
                "add_fingerprint_feature": bool(
                    ensemble_config.add_fingerprint_feature
                ),
                "polynomial_features": ensemble_config.polynomial_features,
                "preprocessor_name": ensemble_config.preprocess_config.name,
                "append_original": ensemble_config.preprocess_config.append_original,
                "global_transformer_name": ensemble_config.preprocess_config.global_transformer_name,
            }
            config_rows.append(row)
            if (
                row["class_permutation"] is not None
                or row["feature_shift_decoder"] is not None
                or row["add_fingerprint_feature"]
                or (row["polynomial_features"] != "no")
                or (row["preprocessor_name"] != "none")
                or (row["append_original"] is not False)
                or (row["global_transformer_name"] is not None)
            ):
                raise RuntimeError(
                    f"An episode violated the preprocessing contract: {row}."
                )
    return {
        "epoch": int(epoch),
        "step": int(step),
        "raw_feature_count": int(expected_feature_count),
        "context_feature_count": observed_context,
        "query_feature_count": observed_query,
        "inferred_categorical_indices": [],
        "post_transform_categorical_indices": flattened_categorical,
        "context_exact_value_and_order_match": bool(
            context_identity["exact_value_and_order_match"]
        ),
        "query_exact_value_and_order_match": bool(
            query_identity["exact_value_and_order_match"]
        ),
        "ensemble_config": config_rows,
        "passed": True,
    }


def _forward_standard_logits(
    classifier: Any, X_query_batch: list[torch.Tensor]
) -> torch.Tensor:
    engine = classifier.executor_
    if len(engine.X_trains) != 1 or len(X_query_batch) != 1:
        raise RuntimeError("clinical adaptation requires batch_size=n_estimators=1.")
    index = 0
    ensemble_config = engine.ensemble_configs[index][0]
    if ensemble_config.class_permutation is not None:
        raise RuntimeError("Class permutation is forbidden in clinical adaptation.")
    model = engine.models[ensemble_config._model_index]
    device = next(model.parameters()).device
    full_x = torch.cat([engine.X_trains[index], X_query_batch[index]], dim=-2)
    context_y = engine.y_trains[index]
    full_x = full_x.to(device)
    context_y = context_y.to(device)
    if engine.force_inference_dtype is not None:
        full_x = full_x.type(engine.force_inference_dtype)
        context_y = context_y.type(engine.force_inference_dtype)
    categorical = [cat_item[index] for cat_item in engine.cat_ix]
    output = model(
        full_x.transpose(0, 1),
        context_y.transpose(0, 1),
        categorical_inds=categorical,
    )
    if output.ndim != 3 or output.shape[1] != 1:
        raise RuntimeError(f"Unexpected TabPFN output shape {tuple(output.shape)}.")
    return output[:, 0, :2]

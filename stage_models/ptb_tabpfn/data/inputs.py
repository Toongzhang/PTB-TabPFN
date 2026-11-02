"""Array input interface; no study feature lists or clinical group mappings."""

from __future__ import annotations
from numbers import Integral
from typing import Any, Mapping
import numpy as np
import pandas as pd
from ptb_tabpfn.evaluation.metrics import validate_binary_labels

STAGES = ("early", "mid", "late")


def _patient_ids(values, count: int) -> np.ndarray:
    raw = np.asarray(values).reshape(-1)
    if len(raw) != count or pd.isna(raw).any():
        raise ValueError("Patient IDs must be nonmissing and align with input rows.")
    ids = raw.astype(str)
    if any(not value.strip() for value in ids) or len(set(ids)) != count:
        raise ValueError(
            "Patient IDs must be nonempty and unique within a stage/split."
        )
    return ids


def _stage_arrays(
    stage: str,
    *,
    X_train,
    y_train,
    train_patient_ids,
    feature_group_ids,
    group_count: int,
    X_validation=None,
    y_validation=None,
    validation_patient_ids=None,
) -> dict[str, Any]:
    if stage not in STAGES:
        raise ValueError(f"Stage must be one of {STAGES}.")
    if (
        isinstance(group_count, bool)
        or not isinstance(group_count, Integral)
        or group_count < 1
    ):
        raise ValueError("group_count must be a positive integer.")
    X = np.asarray(X_train, dtype=np.float32)
    if X.ndim != 2 or not X.shape[0] or not X.shape[1] or np.isinf(X).any():
        raise ValueError(
            "Training features must be a nonempty matrix of finite values or NaN."
        )
    y = validate_binary_labels(y_train)
    if len(y) != len(X):
        raise ValueError("Training features and labels must align.")
    ids = _patient_ids(train_patient_ids, len(X))
    groups = np.asarray(feature_group_ids)
    if (
        groups.ndim != 1
        or groups.dtype.kind not in "iu"
        or len(groups) != X.shape[1]
        or (groups < 0).any()
        or (groups >= group_count).any()
    ):
        raise ValueError(
            "Supply one integer clinical group ID per ordered feature column."
        )
    validation = (X_validation, y_validation, validation_patient_ids)
    if all(value is None for value in validation):
        Xv = np.empty((0, X.shape[1]), dtype=np.float32)
        yv = np.empty(0, dtype=int)
        idv = np.empty(0, dtype=str)
    elif any(value is None for value in validation):
        raise ValueError("Supply validation features, labels and patient IDs together.")
    else:
        Xv = np.asarray(X_validation, dtype=np.float32)
        if Xv.ndim != 2 or Xv.shape[1] != X.shape[1] or np.isinf(Xv).any():
            raise ValueError(
                "Validation columns must match training columns in order and count."
            )
        if len(Xv):
            yv = validate_binary_labels(y_validation)
        else:
            if np.asarray(y_validation).size:
                raise ValueError("Empty validation features require empty labels.")
            yv = np.empty(0, dtype=int)
        if len(yv) != len(Xv):
            raise ValueError("Validation features and labels must align.")
        idv = _patient_ids(validation_patient_ids, len(Xv))
    if set(ids) & set(idv):
        raise ValueError("Training and validation patients must be disjoint.")
    return dict(
        stage=stage,
        X_train=X,
        y_train=y,
        train_patient_ids=ids,
        X_validation=Xv,
        y_validation=yv,
        validation_patient_ids=idv,
        feature_group_ids=groups.astype(int).tolist(),
    )


def make_stage_dataset(
    stage: str,
    *,
    X_train,
    y_train,
    train_patient_ids,
    feature_group_ids,
    group_count: int,
    X_validation=None,
    y_validation=None,
    validation_patient_ids=None,
):
    """Build a StageDataset from caller-prepared arrays and ordered group IDs.

    NaN values and column order are preserved. Omit all validation inputs for
    final training. Features and clinical group assignments are caller-supplied.
    """
    values = _stage_arrays(
        stage,
        X_train=X_train,
        y_train=y_train,
        train_patient_ids=train_patient_ids,
        feature_group_ids=feature_group_ids,
        group_count=group_count,
        X_validation=X_validation,
        y_validation=y_validation,
        validation_patient_ids=validation_patient_ids,
    )
    from ptb_tabpfn.training.joint_training import StageDataset

    return StageDataset(**values)


def validate_stage_data(
    stage_data: Mapping[str, Any], *, group_count: int, require_validation: bool
) -> None:
    """Validate stage alignment, labels and patient-level split consistency."""
    if tuple(stage_data) != STAGES:
        raise ValueError(f"Supply stage_data in the order {STAGES}.")
    registry = {}
    for stage in STAGES:
        data = stage_data[stage]
        if data.stage != stage:
            raise ValueError("Stage key and dataset stage differ.")
        checked = _stage_arrays(
            stage,
            group_count=group_count,
            **{
                key: getattr(data, key)
                for key in (
                    "X_train",
                    "y_train",
                    "train_patient_ids",
                    "feature_group_ids",
                    "X_validation",
                    "y_validation",
                    "validation_patient_ids",
                )
            },
        )
        if bool(len(checked["y_validation"])) != require_validation:
            raise ValueError(
                "Validation rows are required for validation training and absent for final training."
            )
        for split, id_key, label_key in (
            ("train", "train_patient_ids", "y_train"),
            ("val", "validation_patient_ids", "y_validation"),
        ):
            for patient_id, label in zip(
                checked[id_key], checked[label_key], strict=True
            ):
                value = (split, int(label))
                if registry.setdefault(patient_id, value) != value:
                    raise ValueError(
                        "A patient's split and outcome must agree across all available stages."
                    )

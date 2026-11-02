"""Shared binary probability, classification and calibration metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


def validate_binary_labels(y_true, *, require_both: bool = True) -> np.ndarray:
    """Check original labels before integer conversion."""
    if np.iscomplexobj(y_true):
        raise ValueError("Labels must be real binary values.")
    try:
        raw = np.asarray(y_true, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError("Labels must be numeric 0/1 values.") from exc
    if not len(raw) or not np.isfinite(raw).all() or not np.isin(raw, [0, 1]).all():
        raise ValueError("Labels must be nonempty, finite 0/1 values.")
    if require_both and set(np.unique(raw)) != {0, 1}:
        raise ValueError("Evaluation requires both outcome classes.")
    return raw.astype(int)


def validate_binary_inputs(y_true, probabilities) -> tuple[np.ndarray, np.ndarray]:
    y = validate_binary_labels(y_true)
    if np.iscomplexobj(probabilities):
        raise ValueError("Probabilities must be real values.")
    try:
        p = np.asarray(probabilities, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError("Probabilities must be numeric.") from exc
    if len(y) != len(p) or not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError("Probabilities must align with labels and lie in [0, 1].")
    return y, p


def probability_metrics(y_true, probabilities) -> dict[str, float]:
    """AUROC, average precision (AUPRC) and Brier score."""
    y, p = validate_binary_inputs(y_true, probabilities)
    original = np.asarray(probabilities).reshape(-1)
    # Keep float32 prediction scoring identical to the existing ML baselines.
    brier_input = original if original.dtype.kind == "f" else p
    return {
        "auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, brier_input)),
    }


def calibration_intercept_slope(y_true, probabilities) -> dict[str, float | bool]:
    y, p = validate_binary_inputs(y_true, probabilities)
    y = y.astype(float)
    p = np.clip(p, 1e-06, 1 - 1e-06)
    logit = np.log(p / (1.0 - p))
    design = np.column_stack([np.ones(len(y), dtype=float), logit])

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        linear = design @ beta
        loss = float(np.sum(np.logaddexp(0.0, linear) - y * linear))
        fitted = 1.0 / (1.0 + np.exp(-np.clip(linear, -40.0, 40.0)))
        return loss, design.T @ (fitted - y)

    result = minimize(
        lambda beta: objective(beta)[0],
        x0=np.asarray([0.0, 1.0]),
        jac=lambda beta: objective(beta)[1],
        method="BFGS",
        options={"maxiter": 1000, "gtol": 1e-09},
    )
    return {
        "calibration_intercept": float(result.x[0]),
        "calibration_slope": float(result.x[1]),
        "calibration_fit_converged": bool(result.success),
    }


def binary_probability_metrics(y_true, probabilities) -> dict[str, Any]:
    """Training/evaluation callback; retain keys used for checkpoint selection."""
    y, p = validate_binary_inputs(y_true, probabilities)
    values = probability_metrics(y, p)
    return {
        "patient_count": int(len(y)),
        "positive_count": int(y.sum()),
        "positive_rate": float(y.mean()),
        "pr_auc": values["pr_auc"],
        "auc": values["auc"],
        "brier": values["brier"],
        **calibration_intercept_slope(y, p),
    }


__all__ = [
    "probability_metrics",
    "binary_probability_metrics",
    "calibration_intercept_slope",
    "validate_binary_labels",
    "validate_binary_inputs",
]

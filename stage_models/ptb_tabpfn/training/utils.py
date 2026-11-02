"""Shared checkpoint, gradient and parameter-audit helpers."""

from __future__ import annotations
import math
from collections import defaultdict
from typing import Any, Mapping
import torch
from torch import nn


def clone_state_dict_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    """Clone a model state dictionary to CPU for checkpointing/comparison."""
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def load_cpu_state_dict(
    model: nn.Module, state_dict: Mapping[str, torch.Tensor]
) -> None:
    """Load a CPU state dictionary into a model on its current device."""
    current = model.state_dict()
    missing = sorted(set(current) - set(state_dict))
    unexpected = sorted(set(state_dict) - set(current))
    if missing or unexpected:
        raise ValueError(
            f"Checkpoint keys do not match model keys. missing={missing[:5]}, unexpected={unexpected[:5]}."
        )
    converted = {
        name: tensor.to(device=current[name].device, dtype=current[name].dtype)
        for name, tensor in state_dict.items()
    }
    model.load_state_dict(converted, strict=True)


def _parameter_group(name: str) -> str:
    parts = name.split(".")
    if len(parts) >= 3 and parts[0] in {"transformer_encoder", "transformer_decoder"}:
        if parts[1] == "layers" and parts[2].isdigit():
            return ".".join(parts[:3])
    if parts and parts[0] in {
        "encoder",
        "y_encoder",
        "decoder_dict",
        "feature_positional_embedding_embeddings",
    }:
        return parts[0]
    return ".".join(parts[:2]) if len(parts) >= 2 else parts[0]


def parameter_change_rows(
    initial_state: Mapping[str, torch.Tensor], final_state: Mapping[str, torch.Tensor]
) -> list[dict[str, Any]]:
    """Summarize absolute and relative parameter changes by model component."""
    if set(initial_state) != set(final_state):
        raise ValueError("Initial and final state dictionaries have different keys.")
    accumulator: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {
            "parameter_tensor_count": 0,
            "element_count": 0,
            "initial_squared_norm": 0.0,
            "delta_squared_norm": 0.0,
            "changed_element_count": 0,
        }
    )
    for name, initial in initial_state.items():
        final = final_state[name].detach().cpu()
        initial_cpu = initial.detach().cpu()
        if not (initial_cpu.is_floating_point() or initial_cpu.is_complex()):
            continue
        delta = final.to(torch.float64) - initial_cpu.to(torch.float64)
        initial_float = initial_cpu.to(torch.float64)
        group = _parameter_group(name)
        row = accumulator[group]
        row["parameter_tensor_count"] = int(row["parameter_tensor_count"]) + 1
        row["element_count"] = int(row["element_count"]) + int(initial.numel())
        row["initial_squared_norm"] = float(row["initial_squared_norm"]) + float(
            torch.sum(initial_float * initial_float).item()
        )
        row["delta_squared_norm"] = float(row["delta_squared_norm"]) + float(
            torch.sum(delta * delta).item()
        )
        row["changed_element_count"] = int(row["changed_element_count"]) + int(
            torch.count_nonzero(delta).item()
        )
    output: list[dict[str, Any]] = []
    for group, values in sorted(accumulator.items()):
        initial_norm = math.sqrt(float(values["initial_squared_norm"]))
        delta_norm = math.sqrt(float(values["delta_squared_norm"]))
        output.append(
            {
                "parameter_group": group,
                "parameter_tensor_count": int(values["parameter_tensor_count"]),
                "element_count": int(values["element_count"]),
                "changed_element_count": int(values["changed_element_count"]),
                "initial_l2_norm": float(initial_norm),
                "delta_l2_norm": float(delta_norm),
                "relative_delta_l2": float(delta_norm / initial_norm)
                if initial_norm > 0
                else float("nan"),
            }
        )
    return output


def gradient_summary(model: nn.Module) -> dict[str, Any]:
    """Return finite-gradient and norm diagnostics before optimizer stepping."""
    squared_norm = 0.0
    max_abs = 0.0
    tensors_with_grad = 0
    tensors_with_nonzero_grad = 0
    nonfinite_tensors: list[str] = []
    sampled_nonzero_names: list[str] = []
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        tensors_with_grad += 1
        finite = bool(torch.isfinite(gradient).all().item())
        if not finite:
            nonfinite_tensors.append(name)
            continue
        gradient_float = gradient.detach().to(torch.float64)
        squared_norm += float(torch.sum(gradient_float * gradient_float).item())
        if gradient.numel():
            max_abs = max(max_abs, float(torch.max(torch.abs(gradient_float)).item()))
        if bool(torch.count_nonzero(gradient).item()):
            tensors_with_nonzero_grad += 1
            if len(sampled_nonzero_names) < 20:
                sampled_nonzero_names.append(name)
    return {
        "gradient_l2_norm": float(math.sqrt(squared_norm)),
        "gradient_max_abs": float(max_abs),
        "parameter_tensors_with_grad": int(tensors_with_grad),
        "parameter_tensors_with_nonzero_grad": int(tensors_with_nonzero_grad),
        "nonfinite_gradient_tensor_count": int(len(nonfinite_tensors)),
        "nonfinite_gradient_parameter_names": nonfinite_tensors,
        "sample_nonzero_gradient_parameter_names": sampled_nonzero_names,
        "all_observed_gradients_finite": not nonfinite_tensors,
    }


def state_change_audit(
    initial: dict[str, torch.Tensor],
    final: dict[str, torch.Tensor],
    *,
    allowed: set[str],
) -> dict[str, Any]:
    if set(initial) != set(final):
        raise ValueError("Initial and final state dictionaries have different keys.")
    changed = sorted(
        (
            name
            for name in initial
            if (initial[name].is_floating_point() or initial[name].is_complex())
            and (not torch.equal(initial[name], final[name].detach().cpu()))
        )
    )
    unexpected = sorted(set(changed) - allowed)
    return {
        "changed_parameter_names": changed,
        "unexpected_changed_parameter_names": unexpected,
        "allowed_parameter_names_without_change": sorted(allowed - set(changed)),
        "only_allowed_parameters_changed": bool(changed and (not unexpected)),
    }

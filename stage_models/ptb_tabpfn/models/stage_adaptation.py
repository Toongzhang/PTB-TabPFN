"""Shared and stage-specific clinical adaptation for PTB-TabPFN."""

from __future__ import annotations
import copy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import torch
from torch import nn
from torch.nn import functional as F
from ptb_tabpfn.models.clinical_adaptation import (
    _NAN_INDICATOR,
    _as_group_ids,
    _clinical_encoder_class,
    _find_global_missing_linear,
    _token_layout,
)

STAGE_NAMES = ("early", "mid", "late")


class StageClinicalLayout(nn.Module):
    """Non-parametric feature-to-clinical-group layout for one stage."""

    def __init__(
        self,
        feature_group_ids: Sequence[int] | torch.Tensor,
        *,
        group_count: int,
        features_per_token: int,
    ) -> None:
        super().__init__()
        ids = _as_group_ids(feature_group_ids, group_count=group_count)
        token_ids, token_valid = _token_layout(
            ids, features_per_token=features_per_token
        )
        self.feature_count = int(ids.numel())
        self.token_count = int(token_ids.shape[0])
        self.register_buffer("feature_group_ids", ids)
        self.register_buffer("token_group_ids", token_ids)
        self.register_buffer("token_feature_valid", token_valid)


@dataclass(frozen=True)
class StageSharedPrivateInjectionSummary:
    """Audit record for PTB-TabPFN runtime injection."""

    stage_feature_counts: dict[str, int]
    stage_token_counts: dict[str, int]
    group_count: int
    embedding_size: int
    layer_count: int
    features_per_token: int
    replace_global_missing: bool
    symmetric_relations: bool
    alpha_init: float
    beta_init: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "stages": list(STAGE_NAMES),
            "stage_feature_counts": dict(self.stage_feature_counts),
            "stage_token_counts": dict(self.stage_token_counts),
            "group_count": int(self.group_count),
            "embedding_size": int(self.embedding_size),
            "feature_attention_layer_count": int(self.layer_count),
            "features_per_token": int(self.features_per_token),
            "replace_global_missing": bool(self.replace_global_missing),
            "symmetric_relations": bool(self.symmetric_relations),
            "alpha_init": float(self.alpha_init),
            "beta_init": float(self.beta_init),
            "shared_parameters": [
                "slot_group_embedding",
                "slot_missing_embedding",
                "shared_group_relation_delta",
                "clinical_relation_alpha[layer]",
            ],
            "stage_private_parameters": [
                "stage_missing_residual[stage]",
                "stage_group_relation_delta[stage]",
                "clinical_relation_beta[layer, stage]",
            ],
            "input_injection_point": "encoder output before add_embeddings/Transformer",
            "attention_scope": "transformer_encoder.layers[*].self_attn_between_features",
        }


class StageSharedPrivateClinicalRepresentation(nn.Module):
    """Shared clinical representation with small early/mid/late residuals."""

    def __init__(
        self,
        *,
        stage_feature_group_ids: Mapping[str, Sequence[int] | torch.Tensor],
        group_count: int,
        embedding_size: int,
        features_per_token: int,
        relation_prior: torch.Tensor | Sequence[Sequence[float]] | None = None,
        replace_global_missing: bool = True,
        symmetric_relations: bool = True,
    ) -> None:
        super().__init__()
        if tuple(stage_feature_group_ids) != STAGE_NAMES:
            raise ValueError(
                f"stage_feature_group_ids must be ordered exactly as {STAGE_NAMES}, got {tuple(stage_feature_group_ids)}."
            )
        self.group_count = int(group_count)
        self.embedding_size = int(embedding_size)
        self.features_per_token = int(features_per_token)
        self.replace_global_missing = bool(replace_global_missing)
        self.symmetric_relations = bool(symmetric_relations)
        self.stage_layouts = nn.ModuleDict(
            {
                stage: StageClinicalLayout(
                    stage_feature_group_ids[stage],
                    group_count=group_count,
                    features_per_token=features_per_token,
                )
                for stage in STAGE_NAMES
            }
        )
        self.active_stage = STAGE_NAMES[0]
        if relation_prior is None:
            prior = torch.eye(group_count, dtype=torch.float32)
        else:
            prior = torch.as_tensor(relation_prior, dtype=torch.float32)
        if tuple(prior.shape) != (group_count, group_count):
            raise ValueError(
                f"relation_prior must have shape ({group_count}, {group_count}), got {tuple(prior.shape)}."
            )
        if not bool(torch.isfinite(prior).all().item()):
            raise ValueError("relation_prior must contain only finite values.")
        if symmetric_relations:
            prior = 0.5 * (prior + prior.transpose(0, 1))
        self.register_buffer("relation_prior", prior)
        self.slot_group_embedding = nn.Parameter(
            torch.zeros(features_per_token, group_count, embedding_size)
        )
        self.slot_missing_embedding = nn.Parameter(
            torch.zeros(features_per_token, group_count, embedding_size)
        )
        self.shared_group_relation_delta = nn.Parameter(
            torch.zeros(group_count, group_count)
        )
        self.stage_missing_residual = nn.ParameterDict(
            {
                stage: nn.Parameter(
                    torch.zeros(features_per_token, group_count, embedding_size)
                )
                for stage in STAGE_NAMES
            }
        )
        self.stage_group_relation_delta = nn.ParameterDict(
            {
                stage: nn.Parameter(torch.zeros(group_count, group_count))
                for stage in STAGE_NAMES
            }
        )

    def set_active_stage(self, stage: str) -> None:
        if stage not in STAGE_NAMES:
            raise ValueError(f"Unknown stage {stage!r}; expected {STAGE_NAMES}.")
        self.active_stage = str(stage)

    @property
    def active_layout(self) -> StageClinicalLayout:
        return self.stage_layouts[self.active_stage]

    def initialize_missing_from_legacy_encoder(self, encoder: nn.Module) -> None:
        if not self.replace_global_missing:
            return
        linear = _find_global_missing_linear(
            encoder, features_per_token=self.features_per_token
        )
        missing_weights = linear.weight[
            :, self.features_per_token : self.features_per_token * 2
        ]
        initial = _NAN_INDICATOR * missing_weights.detach().transpose(0, 1).unsqueeze(1)
        with torch.no_grad():
            self.slot_missing_embedding.copy_(initial.expand(-1, self.group_count, -1))

    def _symmetrize(self, matrix: torch.Tensor) -> torch.Tensor:
        if self.symmetric_relations:
            return 0.5 * (matrix + matrix.transpose(0, 1))
        return matrix

    def effective_shared_relation(self) -> torch.Tensor:
        delta = self._symmetrize(self.shared_group_relation_delta)
        return self.relation_prior.to(delta) + delta

    def effective_stage_relation_residual(self) -> torch.Tensor:
        return self._symmetrize(self.stage_group_relation_delta[self.active_stage])

    def _relation_to_tokens(
        self,
        relation: torch.Tensor,
        *,
        include_target_token: bool,
        dtype: torch.dtype | None,
        device: torch.device | None,
    ) -> torch.Tensor:
        layout = self.active_layout
        groups = layout.token_group_ids.to(dtype=torch.long)
        valid = layout.token_feature_valid.to(dtype=torch.bool)
        token_count = layout.token_count
        left = groups[:, :, None, None].expand(
            token_count, self.features_per_token, token_count, self.features_per_token
        )
        right = groups[None, None, :, :].expand_as(left)
        pair_values = relation[left, right]
        pair_valid = (valid[:, :, None, None] & valid[None, None, :, :]).to(
            pair_values.dtype
        )
        token_bias = (pair_values * pair_valid).sum(dim=(1, 3)) / float(
            self.features_per_token**2
        )
        if include_target_token:
            output = token_bias.new_zeros(token_count + 1, token_count + 1)
            output[:token_count, :token_count] = token_bias
        else:
            output = token_bias
        if device is not None or dtype is not None:
            output = output.to(device=device, dtype=dtype)
        return output

    def token_relation_bias_components(
        self,
        *,
        include_target_token: bool = True,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self._relation_to_tokens(
                self.effective_shared_relation(),
                include_target_token=include_target_token,
                dtype=dtype,
                device=device,
            ),
            self._relation_to_tokens(
                self.effective_stage_relation_residual(),
                include_target_token=include_target_token,
                dtype=dtype,
                device=device,
            ),
        )

    def _slot_vectors(self) -> tuple[torch.Tensor, torch.Tensor]:
        layout = self.active_layout
        groups = layout.token_group_ids.to(dtype=torch.long)
        slots = torch.arange(self.features_per_token, device=groups.device).unsqueeze(0)
        group_vectors = self.slot_group_embedding[slots, groups]
        missing_vectors = (
            self.slot_missing_embedding + self.stage_missing_residual[self.active_stage]
        )[slots, groups]
        valid = layout.token_feature_valid.to(dtype=torch.bool).unsqueeze(-1)
        return (group_vectors * valid, missing_vectors * valid)

    def add_to_flat_encoder_output(
        self, encoded: torch.Tensor, raw_features: torch.Tensor, *, encoder: nn.Module
    ) -> torch.Tensor:
        if encoded.ndim != 3 or raw_features.ndim != 3:
            raise RuntimeError(
                f"Expected encoded/raw shapes [S, B*T, E] and [S, B*T, k], got {tuple(encoded.shape)} and {tuple(raw_features.shape)}."
            )
        if int(raw_features.shape[-1]) != self.features_per_token:
            raise RuntimeError(
                f"Expected {self.features_per_token} features per token, got {raw_features.shape[-1]}."
            )
        if encoded.shape[:2] != raw_features.shape[:2]:
            raise RuntimeError("Encoded and raw feature axes are misaligned.")
        if int(encoded.shape[-1]) != self.embedding_size:
            raise RuntimeError(
                f"Expected embedding size {self.embedding_size}, got {encoded.shape[-1]}."
            )
        layout = self.active_layout
        merged_batch = int(encoded.shape[1])
        if merged_batch % layout.token_count:
            raise RuntimeError(
                f"Stage {self.active_stage} encoder batch {merged_batch} is not divisible by token count {layout.token_count}."
            )
        if bool(torch.isinf(raw_features).any().item()):
            raise ValueError("Positive/negative infinity is not supported.")
        batch_size = merged_batch // layout.token_count
        missing = torch.isnan(raw_features).to(encoded.dtype)
        group_vectors, missing_vectors = self._slot_vectors()
        group_vectors = group_vectors.to(encoded)
        missing_vectors = missing_vectors.to(encoded)
        group_flat = (
            group_vectors.unsqueeze(0)
            .expand(batch_size, -1, -1, -1)
            .reshape(merged_batch, self.features_per_token, self.embedding_size)
        )
        missing_flat = (
            missing_vectors.unsqueeze(0)
            .expand(batch_size, -1, -1, -1)
            .reshape(merged_batch, self.features_per_token, self.embedding_size)
        )
        clinical_delta = group_flat.sum(dim=1).unsqueeze(0)
        clinical_delta = clinical_delta + (
            missing.unsqueeze(-1) * missing_flat.unsqueeze(0)
        ).sum(dim=2)
        if self.replace_global_missing:
            linear = _find_global_missing_linear(
                encoder, features_per_token=self.features_per_token
            )
            missing_weights = linear.weight[
                :, self.features_per_token : self.features_per_token * 2
            ].to(encoded)
            legacy_delta = F.linear(
                missing * _NAN_INDICATOR, missing_weights, bias=None
            )
            clinical_delta = clinical_delta - legacy_delta
        return encoded + clinical_delta


_STAGE_ATTENTION_CLASS_CACHE: dict[type[nn.Module], type[nn.Module]] = {}


def _stage_clinical_attention_class(base_class: type[nn.Module]) -> type[nn.Module]:
    cached = _STAGE_ATTENTION_CLASS_CACHE.get(base_class)
    if cached is not None:
        return cached
    from tabpfn.architectures.base.memory import support_save_peak_mem_factor

    class StageSharedPrivateFeatureAttention(base_class):
        @support_save_peak_mem_factor
        def _compute(
            self,
            x: torch.Tensor,
            x_kv: torch.Tensor | None,
            k_cache: torch.Tensor | None,
            v_cache: torch.Tensor | None,
            kv_cache: torch.Tensor | None,
            *,
            cache_kv: bool,
            use_cached_kv: bool,
            reuse_first_head_kv: bool,
        ) -> torch.Tensor:
            q, k, v, kv, qkv = self.compute_qkv(
                x,
                x_kv,
                k_cache,
                v_cache,
                kv_cache,
                cache_kv=cache_kv,
                use_cached_kv=use_cached_kv,
                reuse_first_head_kv=reuse_first_head_kv,
            )
            if qkv is not None:
                q, k, v = qkv.unbind(dim=-3)
            elif kv is not None:
                k, v = kv.unbind(dim=-3)
            if q is None or k is None or v is None:
                raise RuntimeError("Stage clinical attention could not construct QKV.")
            if int(q.shape[-2]) != int(k.shape[-2]):
                raise RuntimeError(
                    "Stage clinical feature attention requires equal Q/KV heads."
                )
            adapter = object.__getattribute__(
                self, "_clinical_group_relational_adapter"
            )
            shared_bias, private_bias = adapter.token_relation_bias_components(
                include_target_token=True, dtype=q.dtype, device=q.device
            )
            if int(q.shape[1]) != int(shared_bias.shape[0]) or int(k.shape[1]) != int(
                shared_bias.shape[1]
            ):
                raise RuntimeError(
                    f"Feature-attention sequence length does not match active stage {adapter.active_stage}: q={q.shape[1]}, k={k.shape[1]}, clinical={tuple(shared_bias.shape)}."
                )
            beta = self.clinical_relation_beta[adapter.active_stage]
            additive_bias = (
                (
                    self.clinical_relation_alpha.to(q) * shared_bias
                    + beta.to(q) * private_bias
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )
            extra_inputs: dict[str, Any] = {"attn_mask": additive_bias}
            if self.softmax_scale is not None:
                extra_inputs["scale"] = self.softmax_scale
            attention_heads = self.scaled_dot_product_attention_chunked(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                dropout_p=0.0 if self.dropout_p is None else self.dropout_p,
                **extra_inputs,
            ).transpose(1, 2)
            return torch.einsum("... h d, h d s -> ... s", attention_heads, self._w_out)

    StageSharedPrivateFeatureAttention.__name__ = "StageSharedPrivateFeatureAttention"
    StageSharedPrivateFeatureAttention.__qualname__ = (
        "StageSharedPrivateFeatureAttention"
    )
    StageSharedPrivateFeatureAttention.__module__ = __name__
    _STAGE_ATTENTION_CLASS_CACHE[base_class] = StageSharedPrivateFeatureAttention
    return StageSharedPrivateFeatureAttention


def inject_stage_shared_private_clinical_relational(
    model: nn.Module,
    *,
    stage_feature_group_ids: Mapping[str, Sequence[int] | torch.Tensor],
    group_count: int,
    relation_prior: torch.Tensor | Sequence[Sequence[float]] | None = None,
    replace_global_missing: bool = True,
    symmetric_relations: bool = True,
    alpha_init: float = 0.0,
    beta_init: float = 1.0,
) -> StageSharedPrivateInjectionSummary:
    """Inject PTB-TabPFN into one loaded TabPFN model."""
    if hasattr(model, "clinical_group_relational"):
        raise ValueError("A clinical group-relational adapter already exists.")
    features_per_token = int(getattr(model, "features_per_group", -1))
    embedding_size = int(getattr(model, "ninp", 0))
    if features_per_token <= 0 or embedding_size <= 0:
        raise ValueError("Could not read TabPFN token or embedding width.")
    encoder = getattr(model, "encoder", None)
    layers_obj = getattr(getattr(model, "transformer_encoder", None), "layers", None)
    if not isinstance(encoder, nn.Module) or layers_obj is None:
        raise AttributeError("Could not find TabPFN encoder/layers.")
    layers = list(layers_obj)
    if not layers:
        raise ValueError("TabPFN transformer has no encoder layers.")
    adapter = StageSharedPrivateClinicalRepresentation(
        stage_feature_group_ids=stage_feature_group_ids,
        group_count=group_count,
        embedding_size=embedding_size,
        features_per_token=features_per_token,
        relation_prior=relation_prior,
        replace_global_missing=replace_global_missing,
        symmetric_relations=symmetric_relations,
    )
    reference = next(model.parameters())
    adapter = adapter.to(device=reference.device, dtype=reference.dtype)
    adapter.initialize_missing_from_legacy_encoder(encoder)
    model.add_module("clinical_group_relational", adapter)
    encoder.__class__ = _clinical_encoder_class(encoder.__class__)
    object.__setattr__(encoder, "_clinical_group_relational_adapter", adapter)
    for layer_index, layer in enumerate(layers):
        attention = getattr(layer, "self_attn_between_features", None)
        if not isinstance(attention, nn.Module):
            raise AttributeError(f"Layer {layer_index} has no feature attention.")
        if int(getattr(attention, "_nhead", -1)) != int(
            getattr(attention, "_nhead_kv", -2)
        ):
            raise ValueError(f"Layer {layer_index} has unequal query/KV head counts.")
        attention.__class__ = _stage_clinical_attention_class(attention.__class__)
        attention_reference = next(attention.parameters())
        attention.register_parameter(
            "clinical_relation_alpha",
            nn.Parameter(
                torch.tensor(
                    alpha_init,
                    device=attention_reference.device,
                    dtype=attention_reference.dtype,
                )
            ),
        )
        attention.add_module(
            "clinical_relation_beta",
            nn.ParameterDict(
                {
                    stage: nn.Parameter(
                        torch.tensor(
                            beta_init,
                            device=attention_reference.device,
                            dtype=attention_reference.dtype,
                        )
                    )
                    for stage in STAGE_NAMES
                }
            ),
        )
        object.__setattr__(attention, "_clinical_group_relational_adapter", adapter)
    return StageSharedPrivateInjectionSummary(
        stage_feature_counts={
            stage: adapter.stage_layouts[stage].feature_count for stage in STAGE_NAMES
        },
        stage_token_counts={
            stage: adapter.stage_layouts[stage].token_count for stage in STAGE_NAMES
        },
        group_count=int(group_count),
        embedding_size=embedding_size,
        layer_count=len(layers),
        features_per_token=features_per_token,
        replace_global_missing=replace_global_missing,
        symmetric_relations=symmetric_relations,
        alpha_init=float(alpha_init),
        beta_init=float(beta_init),
    )


def set_active_clinical_stage(model: nn.Module, stage: str) -> None:
    adapter = getattr(model, "clinical_group_relational", None)
    if not isinstance(adapter, StageSharedPrivateClinicalRepresentation):
        raise RuntimeError("Model has no PTB-TabPFN adapter.")
    adapter.set_active_stage(stage)


def stage_shared_private_parameter_names(model: nn.Module) -> set[str]:
    return {
        name
        for name, _ in model.named_parameters()
        if name.startswith("clinical_group_relational.")
        or name.endswith(".self_attn_between_features.clinical_relation_alpha")
        or ".self_attn_between_features.clinical_relation_beta." in name
    }


def stage_shared_private_parameter_audit(model: nn.Module) -> dict[str, Any]:
    new_names = stage_shared_private_parameter_names(model)
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    shared_names = sorted(
        (
            name
            for name in new_names
            if "stage_missing_residual." not in name
            and "stage_group_relation_delta." not in name
            and ("clinical_relation_beta." not in name)
        )
    )
    private_names = sorted(new_names - set(shared_names))
    return {
        "new_parameter_names": sorted(new_names),
        "shared_parameter_names": shared_names,
        "stage_private_parameter_names": private_names,
        "new_parameter_tensor_count": int(len(new_names)),
        "new_parameter_count": int(
            sum(
                (
                    parameter.numel()
                    for name, parameter in model.named_parameters()
                    if name in new_names
                )
            )
        ),
        "trainable_parameter_names": sorted(trainable_names),
        "unexpected_trainable_parameter_names": sorted(trainable_names - new_names),
        "missing_trainable_parameter_names": sorted(new_names - trainable_names),
        "only_stage_shared_private_parameters_trainable": bool(
            trainable_names and trainable_names == new_names
        ),
    }


def freeze_except_stage_shared_private(model: nn.Module) -> dict[str, Any]:
    allowed = stage_shared_private_parameter_names(model)
    if not allowed:
        raise RuntimeError("No PTB-TabPFN parameters were found.")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in allowed)
    return stage_shared_private_parameter_audit(model)


def stage_private_gradient_routing_audit(
    model: nn.Module, *, active_stage: str
) -> dict[str, Any]:
    if active_stage not in STAGE_NAMES:
        raise ValueError(active_stage)
    foreign_with_nonzero_grad: list[str] = []
    active_with_nonzero_grad: list[str] = []
    nonfinite: list[str] = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if not bool(torch.isfinite(parameter.grad).all().item()):
            nonfinite.append(name)
        nonzero = bool(torch.count_nonzero(parameter.grad).item())
        if not nonzero:
            continue
        is_private = (
            "stage_missing_residual." in name
            or "stage_group_relation_delta." in name
            or "clinical_relation_beta." in name
        )
        if not is_private:
            continue
        if f".{active_stage}" in name:
            active_with_nonzero_grad.append(name)
        else:
            foreign_with_nonzero_grad.append(name)
    return {
        "active_stage": active_stage,
        "active_private_parameters_with_nonzero_grad": sorted(active_with_nonzero_grad),
        "foreign_private_parameters_with_nonzero_grad": sorted(
            foreign_with_nonzero_grad
        ),
        "nonfinite_gradient_parameters": sorted(nonfinite),
        "routing_passed": bool(
            active_with_nonzero_grad
            and (not foreign_with_nonzero_grad)
            and (not nonfinite)
        ),
    }


def assert_stage_shared_private_clone_integrity(model: nn.Module) -> None:
    clone = copy.deepcopy(model)
    adapter = getattr(clone, "clinical_group_relational", None)
    if not isinstance(adapter, StageSharedPrivateClinicalRepresentation):
        raise RuntimeError("Deep-copied model lost its PTB-TabPFN adapter.")
    if (
        object.__getattribute__(clone.encoder, "_clinical_group_relational_adapter")
        is not adapter
    ):
        raise RuntimeError("Deep-copied encoder points to the wrong adapter.")
    for layer_index, layer in enumerate(clone.transformer_encoder.layers):
        if (
            object.__getattribute__(
                layer.self_attn_between_features, "_clinical_group_relational_adapter"
            )
            is not adapter
        ):
            raise RuntimeError(
                f"Layer {layer_index} points to the wrong cloned adapter."
            )


__all__ = [
    "STAGE_NAMES",
    "StageClinicalLayout",
    "StageSharedPrivateClinicalRepresentation",
    "StageSharedPrivateInjectionSummary",
    "assert_stage_shared_private_clone_integrity",
    "freeze_except_stage_shared_private",
    "inject_stage_shared_private_clinical_relational",
    "set_active_clinical_stage",
    "stage_private_gradient_routing_audit",
    "stage_shared_private_parameter_audit",
    "stage_shared_private_parameter_names",
]

"""Clinical group representations, missingness embeddings and feature-relation bias."""

from __future__ import annotations
import copy
import math
from dataclasses import dataclass
from typing import Any, Sequence
import torch
from torch import nn
from torch.nn import functional as F

PROPOSED_FEATURES_PER_TOKEN = 2
_NAN_INDICATOR = -2.0


@dataclass(frozen=True)
class ClinicalGroupRelationalInjectionSummary:
    """Audit record for a Clinical Group-Relational TabPFN injection."""

    feature_count: int
    token_count: int
    group_count: int
    embedding_size: int
    layer_count: int
    features_per_token: int
    replace_global_missing: bool
    symmetric_relations: bool
    alpha_init: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature_count": int(self.feature_count),
            "features_per_token": int(self.features_per_token),
            "token_count": int(self.token_count),
            "group_count": int(self.group_count),
            "embedding_size": int(self.embedding_size),
            "feature_attention_layer_count": int(self.layer_count),
            "replace_global_missing": bool(self.replace_global_missing),
            "symmetric_relations": bool(self.symmetric_relations),
            "alpha_init": float(self.alpha_init),
            "input_injection_point": "encoder output before add_embeddings/Transformer",
            "attention_scope": "transformer_encoder.layers[*].self_attn_between_features",
        }


def _as_group_ids(
    feature_group_ids: Sequence[int] | torch.Tensor, *, group_count: int
) -> torch.Tensor:
    ids = torch.as_tensor(feature_group_ids, dtype=torch.long).reshape(-1)
    if ids.numel() == 0:
        raise ValueError("feature_group_ids cannot be empty.")
    if group_count <= 0:
        raise ValueError("group_count must be positive.")
    if bool((ids < 0).any().item()) or bool((ids >= group_count).any().item()):
        raise ValueError(
            f"Every feature group ID must be in [0, {group_count - 1}], got {ids.tolist()}."
        )
    return ids


def _token_layout(
    feature_group_ids: torch.Tensor, *, features_per_token: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return padded ``[token, slot]`` group IDs and a real-feature mask."""
    feature_count = int(feature_group_ids.numel())
    if features_per_token <= 0:
        raise ValueError("features_per_token must be positive.")
    token_count = math.ceil(feature_count / features_per_token)
    padded_count = token_count * features_per_token
    padded = torch.zeros(padded_count, dtype=torch.long)
    valid = torch.zeros(padded_count, dtype=torch.bool)
    padded[:feature_count] = feature_group_ids
    valid[:feature_count] = True
    return (
        padded.reshape(token_count, features_per_token),
        valid.reshape(token_count, features_per_token),
    )


def _find_global_missing_linear(
    encoder: nn.Module, *, features_per_token: int
) -> nn.Linear:
    candidates: list[nn.Linear] = []
    for module in encoder.modules():
        in_keys = tuple(getattr(module, "in_keys", ()))
        layer = getattr(module, "layer", None)
        if in_keys == ("main", "nan_indicators") and isinstance(layer, nn.Linear):
            candidates.append(layer)
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one TabPFN linear input encoder consuming ('main', 'nan_indicators'); found {len(candidates)}. The replacement path supports the v2/v2.5 linear encoder only."
        )
    linear = candidates[0]
    expected_inputs = features_per_token * 2
    if int(linear.in_features) != expected_inputs:
        raise RuntimeError(
            f"Expected input encoder width {expected_inputs}, got {linear.in_features}."
        )
    return linear


class ClinicalGroupRepresentation(nn.Module):
    """Shared clinical input embeddings and group-relation parameters."""

    def __init__(
        self,
        *,
        feature_group_ids: Sequence[int] | torch.Tensor,
        group_count: int,
        embedding_size: int,
        features_per_token: int = PROPOSED_FEATURES_PER_TOKEN,
        relation_prior: torch.Tensor | Sequence[Sequence[float]] | None = None,
        replace_global_missing: bool = True,
        symmetric_relations: bool = True,
    ) -> None:
        super().__init__()
        ids = _as_group_ids(feature_group_ids, group_count=group_count)
        token_ids, token_valid = _token_layout(
            ids, features_per_token=features_per_token
        )
        self.feature_count = int(ids.numel())
        self.token_count = int(token_ids.shape[0])
        self.group_count = int(group_count)
        self.embedding_size = int(embedding_size)
        self.features_per_token = int(features_per_token)
        self.replace_global_missing = bool(replace_global_missing)
        self.symmetric_relations = bool(symmetric_relations)
        self.register_buffer("feature_group_ids", ids)
        self.register_buffer("token_group_ids", token_ids)
        self.register_buffer("token_feature_valid", token_valid)
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
            torch.zeros(self.features_per_token, group_count, embedding_size)
        )
        self.slot_missing_embedding = nn.Parameter(
            torch.zeros(self.features_per_token, group_count, embedding_size)
        )
        self.group_relation_delta = nn.Parameter(torch.zeros(group_count, group_count))

    def initialize_missing_from_legacy_encoder(self, encoder: nn.Module) -> None:
        """Copy the shared TabPFN NaN projection into every clinical group.

        The subsequent forward path subtracts the legacy projection.  Copying it
        into every group makes the injected model match the pretrained model at
        initialization for ordinary NaN missingness, while allowing groups to
        diverge during fine-tuning.
        """
        if not self.replace_global_missing:
            return
        linear = _find_global_missing_linear(
            encoder, features_per_token=self.features_per_token
        )
        missing_weights = linear.weight[
            :, self.features_per_token : self.features_per_token * 2
        ]
        if tuple(missing_weights.shape) != (
            self.embedding_size,
            self.features_per_token,
        ):
            raise RuntimeError(
                f"TabPFN input-encoder shape is incompatible with clinical missing embeddings: {tuple(missing_weights.shape)}."
            )
        initial = _NAN_INDICATOR * missing_weights.detach().transpose(0, 1).unsqueeze(1)
        with torch.no_grad():
            self.slot_missing_embedding.copy_(initial.expand(-1, self.group_count, -1))

    def effective_group_relation(self) -> torch.Tensor:
        delta = self.group_relation_delta
        if self.symmetric_relations:
            delta = 0.5 * (delta + delta.transpose(0, 1))
        return self.relation_prior.to(delta) + delta

    def token_relation_bias(
        self,
        *,
        include_target_token: bool = True,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Build ``R_tu`` with the exact fixed ``1 / k^2`` denominator."""
        relation = self.effective_group_relation()
        groups = self.token_group_ids.to(dtype=torch.long)
        valid = self.token_feature_valid.to(dtype=torch.bool)
        token_count = self.token_count
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

    def _slot_vectors(self) -> tuple[torch.Tensor, torch.Tensor]:
        slots = torch.arange(
            self.features_per_token, device=self.token_group_ids.device
        ).unsqueeze(0)
        token_group_ids = self.token_group_ids.to(dtype=torch.long)
        group_vectors = self.slot_group_embedding[slots, token_group_ids]
        missing_vectors = self.slot_missing_embedding[slots, token_group_ids]
        valid = self.token_feature_valid.to(dtype=torch.bool).unsqueeze(-1)
        return (group_vectors * valid, missing_vectors * valid)

    def add_to_flat_encoder_output(
        self, encoded: torch.Tensor, raw_features: torch.Tensor, *, encoder: nn.Module
    ) -> torch.Tensor:
        """Add clinical representations to ``[S, B*T, E]`` encoder output."""
        if encoded.ndim != 3 or raw_features.ndim != 3:
            raise RuntimeError(
                f"Expected encoded/raw shapes [S, B*T, E] and [S, B*T, k], got {tuple(encoded.shape)} and {tuple(raw_features.shape)}."
            )
        if int(raw_features.shape[-1]) != self.features_per_token:
            raise RuntimeError(
                f"Clinical injection requires {self.features_per_token} features per token, got {raw_features.shape[-1]}."
            )
        if encoded.shape[:2] != raw_features.shape[:2]:
            raise RuntimeError("Encoded and raw feature axes are misaligned.")
        if int(encoded.shape[-1]) != self.embedding_size:
            raise RuntimeError(
                f"Expected embedding size {self.embedding_size}, got {encoded.shape[-1]}."
            )
        merged_batch = int(encoded.shape[1])
        if merged_batch % self.token_count:
            raise RuntimeError(
                f"Merged encoder batch {merged_batch} is not divisible by token count {self.token_count}; clinical group IDs are not aligned with the preprocessed feature matrix."
            )
        batch_size = merged_batch // self.token_count
        if bool(torch.isinf(raw_features).any().item()):
            raise ValueError(
                "Clinical missingness is defined as a binary NaN indicator; positive/negative infinity is not supported."
            )
        missing = torch.isnan(raw_features).to(encoded.dtype)
        group_vectors, missing_vectors = self._slot_vectors()
        group_vectors = group_vectors.to(device=encoded.device, dtype=encoded.dtype)
        missing_vectors = missing_vectors.to(device=encoded.device, dtype=encoded.dtype)
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
            ].to(device=encoded.device, dtype=encoded.dtype)
            legacy_indicators = missing * _NAN_INDICATOR
            legacy_delta = F.linear(legacy_indicators, missing_weights, bias=None)
            clinical_delta = clinical_delta - legacy_delta
        return encoded + clinical_delta


_ENCODER_CLASS_CACHE: dict[type[nn.Module], type[nn.Module]] = {}
_ATTENTION_CLASS_CACHE: dict[type[nn.Module], type[nn.Module]] = {}


def _clinical_encoder_class(base_class: type[nn.Module]) -> type[nn.Module]:
    cached = _ENCODER_CLASS_CACHE.get(base_class)
    if cached is not None:
        return cached
    base_forward = base_class.forward

    class ClinicalGroupInputEncoder(base_class):
        """Runtime encoder subclass that adds clinical token representations."""

        def forward(self, input: dict[str, torch.Tensor], **kwargs: Any):
            if not isinstance(input, dict) or "main" not in input:
                raise RuntimeError(
                    "Clinical TabPFN encoder expects an input dict with key 'main'."
                )
            raw_features = input["main"]
            encoded = base_forward(self, input, **kwargs)
            adapter = object.__getattribute__(
                self, "_clinical_group_relational_adapter"
            )
            return adapter.add_to_flat_encoder_output(
                encoded, raw_features, encoder=self
            )

    ClinicalGroupInputEncoder.__name__ = "ClinicalGroupInputEncoder"
    ClinicalGroupInputEncoder.__qualname__ = "ClinicalGroupInputEncoder"
    ClinicalGroupInputEncoder.__module__ = __name__
    _ENCODER_CLASS_CACHE[base_class] = ClinicalGroupInputEncoder
    return ClinicalGroupInputEncoder


def _clinical_attention_class(base_class: type[nn.Module]) -> type[nn.Module]:
    cached = _ATTENTION_CLASS_CACHE.get(base_class)
    if cached is not None:
        return cached
    from tabpfn.architectures.base.memory import support_save_peak_mem_factor

    class ClinicalRelationFeatureAttention(base_class):
        """Runtime feature-attention subclass with additive clinical bias."""

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
                raise RuntimeError(
                    "Clinical feature attention could not construct QKV."
                )
            if int(q.shape[-2]) != int(k.shape[-2]):
                raise RuntimeError(
                    "Clinical feature attention currently requires equal Q and KV head counts."
                )
            adapter = object.__getattribute__(
                self, "_clinical_group_relational_adapter"
            )
            relation_bias = adapter.token_relation_bias(
                include_target_token=True, dtype=q.dtype, device=q.device
            )
            if int(q.shape[1]) != int(relation_bias.shape[0]) or int(k.shape[1]) != int(
                relation_bias.shape[1]
            ):
                raise RuntimeError(
                    f"Feature-attention sequence length does not match the clinical token layout. This usually means TabPFN preprocessing changed the feature count or order: q={q.shape[1]}, k={k.shape[1]}, clinical={tuple(relation_bias.shape)}."
                )
            additive_bias = (
                (self.clinical_relation_alpha.to(q) * relation_bias)
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

    ClinicalRelationFeatureAttention.__name__ = "ClinicalRelationFeatureAttention"
    ClinicalRelationFeatureAttention.__qualname__ = "ClinicalRelationFeatureAttention"
    ClinicalRelationFeatureAttention.__module__ = __name__
    _ATTENTION_CLASS_CACHE[base_class] = ClinicalRelationFeatureAttention
    return ClinicalRelationFeatureAttention


def inject_clinical_group_relational(
    model: nn.Module,
    *,
    feature_group_ids: Sequence[int] | torch.Tensor,
    group_count: int,
    relation_prior: torch.Tensor | Sequence[Sequence[float]] | None = None,
    replace_global_missing: bool = True,
    symmetric_relations: bool = True,
    alpha_init: float = 0.0,
) -> ClinicalGroupRelationalInjectionSummary:
    """Inject clinical inputs and relation bias into a loaded TabPFN model."""
    if hasattr(model, "clinical_group_relational"):
        raise ValueError("Clinical group-relational components were already injected.")
    features_per_token = int(getattr(model, "features_per_group", -1))
    if features_per_token <= 0:
        raise ValueError(
            f"Could not determine a positive TabPFN features_per_group value; got {getattr(model, 'features_per_group', None)!r}."
        )
    encoder = getattr(model, "encoder", None)
    if not isinstance(encoder, nn.Module):
        raise AttributeError("Could not find model.encoder.")
    embedding_size = int(getattr(model, "ninp", 0))
    if embedding_size <= 0:
        raise AttributeError("Could not determine model.ninp embedding size.")
    transformer_encoder = getattr(model, "transformer_encoder", None)
    layers_obj = getattr(transformer_encoder, "layers", None)
    if layers_obj is None:
        raise AttributeError("Could not find model.transformer_encoder.layers.")
    layers = list(layers_obj)
    if not layers:
        raise ValueError("TabPFN transformer has no encoder layers.")
    adapter = ClinicalGroupRepresentation(
        feature_group_ids=feature_group_ids,
        group_count=int(group_count),
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
            raise AttributeError(
                f"Layer {layer_index} has no self_attn_between_features module."
            )
        if int(getattr(attention, "_nhead", -1)) != int(
            getattr(attention, "_nhead_kv", -2)
        ):
            raise ValueError(
                f"clinical adaptation feature attention requires equal query and KV head counts; layer {layer_index} is incompatible."
            )
        attention.__class__ = _clinical_attention_class(attention.__class__)
        reference_attention = next(attention.parameters())
        attention.register_parameter(
            "clinical_relation_alpha",
            nn.Parameter(
                torch.tensor(
                    float(alpha_init),
                    device=reference_attention.device,
                    dtype=reference_attention.dtype,
                )
            ),
        )
        object.__setattr__(attention, "_clinical_group_relational_adapter", adapter)
    summary = ClinicalGroupRelationalInjectionSummary(
        feature_count=adapter.feature_count,
        token_count=adapter.token_count,
        group_count=adapter.group_count,
        embedding_size=adapter.embedding_size,
        layer_count=len(layers),
        features_per_token=adapter.features_per_token,
        replace_global_missing=replace_global_missing,
        symmetric_relations=symmetric_relations,
        alpha_init=float(alpha_init),
    )
    return summary


def clinical_group_relational_parameter_names(model: nn.Module) -> set[str]:
    """Return exactly the parameters newly introduced by clinical adaptation."""
    return {
        name
        for name, _ in model.named_parameters()
        if name.startswith("clinical_group_relational.")
        or name.endswith(".self_attn_between_features.clinical_relation_alpha")
    }


def clinical_group_relational_parameter_audit(model: nn.Module) -> dict[str, Any]:
    new_names = clinical_group_relational_parameter_names(model)
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    return {
        "new_parameter_names": sorted(new_names),
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
        "unexpected_trainable_parameter_names_for_clinical_only": sorted(
            trainable_names - new_names
        ),
        "missing_trainable_clinical_parameter_names": sorted(
            new_names - trainable_names
        ),
        "only_clinical_parameters_trainable": bool(
            trainable_names and trainable_names == new_names
        ),
    }


def freeze_except_clinical_group_relational(model: nn.Module) -> dict[str, Any]:
    """Freeze the pretrained backbone and train only clinical adaptation parameters."""
    allowed = clinical_group_relational_parameter_names(model)
    if not allowed:
        raise RuntimeError("No clinical adaptation parameters were found.")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in allowed)
    return clinical_group_relational_parameter_audit(model)


def assert_clinical_clone_integrity(model: nn.Module) -> None:
    """Verify that deepcopy keeps encoder/attention references on the cloned adapter."""
    clone = copy.deepcopy(model)
    adapter = getattr(clone, "clinical_group_relational", None)
    if adapter is None:
        raise RuntimeError("Deep-copied model lost its clinical adapter.")
    encoder_adapter = object.__getattribute__(
        clone.encoder, "_clinical_group_relational_adapter"
    )
    if encoder_adapter is not adapter:
        raise RuntimeError("Deep-copied encoder points to the wrong clinical adapter.")
    for layer_index, layer in enumerate(clone.transformer_encoder.layers):
        attention_adapter = object.__getattribute__(
            layer.self_attn_between_features, "_clinical_group_relational_adapter"
        )
        if attention_adapter is not adapter:
            raise RuntimeError(
                f"Deep-copied feature attention layer {layer_index} points to the wrong clinical adapter."
            )


__all__ = [
    "PROPOSED_FEATURES_PER_TOKEN",
    "ClinicalGroupRelationalInjectionSummary",
    "ClinicalGroupRepresentation",
    "assert_clinical_clone_integrity",
    "clinical_group_relational_parameter_audit",
    "clinical_group_relational_parameter_names",
    "freeze_except_clinical_group_relational",
    "inject_clinical_group_relational",
]

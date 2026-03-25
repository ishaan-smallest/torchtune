# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
from collections import defaultdict

import torch

from torchtune.models.convert_weights import get_mapped_key

# NOTE: This file is the same as the Qwen2 _convert_weights.py file with one key difference.
# For tied-embedding Qwen2 models, only the embedding weight is stored on the HF Hub.
# However, for Qwen3, both the embedding and output weights are stored on the Hub.
# While we handle the tying ourselves on load, we do need to duplicate the weight to save in HF's format.
# The exception is for Qwen3 4B, which matches the behavior of Qwen2.

# state dict key mappings from HF's format to torchtune's format
_FROM_HF = {
    "model.embed_tokens.weight": "tok_embeddings.weight",
    "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attn.q_proj.weight",
    "model.layers.{}.self_attn.q_proj.bias": "layers.{}.attn.q_proj.bias",
    "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attn.k_proj.weight",
    "model.layers.{}.self_attn.k_proj.bias": "layers.{}.attn.k_proj.bias",
    "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attn.v_proj.weight",
    "model.layers.{}.self_attn.v_proj.bias": "layers.{}.attn.v_proj.bias",
    "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attn.output_proj.weight",
    "model.layers.{}.self_attn.q_norm.weight": "layers.{}.attn.q_norm.scale",
    "model.layers.{}.self_attn.k_norm.weight": "layers.{}.attn.k_norm.scale",
    "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
    "model.layers.{}.mlp.gate_proj.weight": "layers.{}.mlp.w1.weight",
    "model.layers.{}.mlp.up_proj.weight": "layers.{}.mlp.w3.weight",
    "model.layers.{}.mlp.down_proj.weight": "layers.{}.mlp.w2.weight",
    "model.layers.{}.input_layernorm.weight": "layers.{}.sa_norm.scale",
    "model.layers.{}.post_attention_layernorm.weight": "layers.{}.mlp_norm.scale",
    "model.norm.weight": "norm.scale",
    "lm_head.weight": "output.weight",
}


QWEN3_TIED_KEY = "lm_head.weight"
QWEN3_TUNE_EMBEDDING_KEY = "tok_embeddings.weight"


def qwen3_hf_to_tune(
    state_dict: dict[str, torch.Tensor],
    num_heads: int = 32,
    num_kv_heads: int = 32,
    dim: int = 4096,
    head_dim: int = None,
    tie_word_embeddings: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Convert a state dict from HF's format to TorchTune's format, which contains the weights
    of a Qwen3 model.
    State dicts from multiple checkpoint files should be consolidated into a single state dict
    before calling this function.
    The logic is identical to :func:`~torchtune.models.convert_weights.hf_to_tune`, but may not load
    output projection weights.

    Args:
        state_dict (dict[str, torch.Tensor]): State dict in HF's format.
        num_heads (int): Number of heads in the model.
        num_kv_heads (int): Number of heads in the key/value projection layers.
        dim (int): Dimension of the model.
        head_dim (int): Dimension of the head. If not provided, it will be calculated
            as dim // num_heads.
        tie_word_embeddings (bool): Whether the model's input and output word embeddings should be tied.

    Returns:
        dict[str, torch.Tensor]: State dict in torchtune's format.
    """
    converted_state_dict = {}
    if head_dim is None:
        head_dim = dim // num_heads

    for key, value in state_dict.items():
        if (
            tie_word_embeddings and QWEN3_TIED_KEY in key
        ):  # Skip loading the output projection weights
            continue
        if "rotary_emb.inv_freq" in key:  # Skip loading the position embeddings
            continue

        new_key = get_mapped_key(key, _FROM_HF)
        converted_state_dict[new_key] = value
    return converted_state_dict


def qwen3_tune_to_hf(
    state_dict: dict[str, torch.Tensor],
    num_heads: int = 32,
    num_kv_heads: int = 32,
    dim: int = 4096,
    head_dim: int = None,
    tie_word_embeddings: bool = False,
):
    """
    Convert a state dict from torchtune's format to HF's format. This function
    doesn't handle any sharding or splitting of state dicts. It follows the
    state_dict IN -> state_dict OUT pattern.

    Args:
        state_dict (dict[str, torch.Tensor]): State dict in torchtune's format.
        num_heads (int): Number of heads in the model.
        num_kv_heads (int): Number of heads in the key/value projection layers.
        dim (int): Dimension of the model.
        head_dim (int): Dimension of the head. If not provided, it will be calculated
            as dim // num_heads.
        tie_word_embeddings (bool): Whether the model's input and output word embeddings should be tied.

    Returns:
        dict[str, torch.Tensor]: State dict in HF's format.
    """
    converted_state_dict = {}
    inverted_mapping_dict = {v: k for k, v in _FROM_HF.items()}
    if head_dim is None:
        head_dim = dim // num_heads

    for key, value in state_dict.items():
        new_key = get_mapped_key(key, inverted_mapping_dict)
        converted_state_dict[new_key] = value
        if QWEN3_TUNE_EMBEDDING_KEY in key and tie_word_embeddings:
            # If the model's input and output word embeddings are tied, we need to
            # copy the input word embeddings to the output word embeddings
            converted_state_dict["lm_head.weight"] = value.detach().clone()

    return converted_state_dict


# ---------------------------------------------------------------------------
# Qwen3 MoE weight conversion
# ---------------------------------------------------------------------------

# Mapping for non-MoE keys in Qwen3 MoE models (attention, norms, embeddings).
# MoE-specific keys (router, experts) are handled separately because they have
# two numeric indices (layer + expert) which get_mapped_key cannot handle.
_FROM_HF_MOE = {
    "model.embed_tokens.weight": "tok_embeddings.weight",
    "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attn.q_proj.weight",
    "model.layers.{}.self_attn.q_proj.bias": "layers.{}.attn.q_proj.bias",
    "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attn.k_proj.weight",
    "model.layers.{}.self_attn.k_proj.bias": "layers.{}.attn.k_proj.bias",
    "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attn.v_proj.weight",
    "model.layers.{}.self_attn.v_proj.bias": "layers.{}.attn.v_proj.bias",
    "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attn.output_proj.weight",
    "model.layers.{}.self_attn.q_norm.weight": "layers.{}.attn.q_norm.scale",
    "model.layers.{}.self_attn.k_norm.weight": "layers.{}.attn.k_norm.scale",
    "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
    "model.layers.{}.input_layernorm.weight": "layers.{}.sa_norm.scale",
    "model.layers.{}.post_attention_layernorm.weight": "layers.{}.mlp_norm.scale",
    "model.norm.weight": "norm.scale",
    "lm_head.weight": "output.weight",
}

# Regex to match per-expert weight keys in HF format:
#   model.layers.{L}.mlp.experts.{E}.{gate_proj|up_proj|down_proj}.weight
_EXPERT_KEY_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
)

# Regex to match router gate keys:
#   model.layers.{L}.mlp.gate.weight
_ROUTER_KEY_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.gate\.weight$"
)


def qwen3_moe_hf_to_tune(
    state_dict: dict[str, torch.Tensor],
    num_heads: int = 32,
    num_kv_heads: int = 4,
    dim: int = 2048,
    head_dim: int = None,
    tie_word_embeddings: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Convert a state dict from HF's format to torchtune's format for Qwen3 MoE models.

    HF stores each expert as individual tensors with shape ``[out_dim, in_dim]``
    (nn.Linear convention). Torchtune's ``GroupedExperts`` stores them as stacked
    3D parameters with shape ``[num_experts, in_dim, out_dim]``.

    This function:
    1. Collects all per-expert weights grouped by (layer, projection).
    2. Transposes each from ``[out, in]`` to ``[in, out]``.
    3. Stacks them into ``[num_experts, in_dim, out_dim]`` tensors.

    Args:
        state_dict (dict[str, torch.Tensor]): HF state dict (consolidated from shards).
        num_heads (int): Number of attention heads.
        num_kv_heads (int): Number of KV heads.
        dim (int): Hidden dimension.
        head_dim (int): Head dimension.
        tie_word_embeddings (bool): Whether embeddings are tied.

    Returns:
        dict[str, torch.Tensor]: State dict in torchtune's format.
    """
    converted = {}
    if head_dim is None:
        head_dim = dim // num_heads

    # Accumulate per-expert weights: (layer_idx, proj_name) -> {expert_idx: tensor}
    expert_weights: dict[tuple[int, str], dict[int, torch.Tensor]] = defaultdict(dict)

    for key, value in state_dict.items():
        if "rotary_emb.inv_freq" in key:
            continue
        if tie_word_embeddings and key == QWEN3_TIED_KEY:
            continue

        # Check for per-expert weight
        m = _EXPERT_KEY_RE.match(key)
        if m:
            layer_idx = int(m.group(1))
            expert_idx = int(m.group(2))
            proj_name = m.group(3)  # gate_proj, up_proj, or down_proj
            # Transpose from HF's [out_dim, in_dim] to torchtune's [in_dim, out_dim]
            expert_weights[(layer_idx, proj_name)][expert_idx] = value.t()
            continue

        # Check for router gate
        m = _ROUTER_KEY_RE.match(key)
        if m:
            layer_idx = int(m.group(1))
            # Router gate is nn.Linear, stored as [num_experts, dim] in both formats
            converted[f"layers.{layer_idx}.mlp.router.gate.weight"] = value
            continue

        # Standard attention / norm / embedding keys
        new_key = get_mapped_key(key, _FROM_HF_MOE)
        converted[new_key] = value

    # Stack expert weights into 3D tensors
    for (layer_idx, proj_name), experts_dict in expert_weights.items():
        num_experts = max(experts_dict.keys()) + 1
        stacked = torch.stack([experts_dict[i] for i in range(num_experts)])
        converted[f"layers.{layer_idx}.mlp.experts.{proj_name}"] = stacked

    return converted


def qwen3_moe_tune_to_hf(
    state_dict: dict[str, torch.Tensor],
    num_heads: int = 32,
    num_kv_heads: int = 4,
    dim: int = 2048,
    head_dim: int = None,
    tie_word_embeddings: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Convert a state dict from torchtune's format to HF's format for Qwen3 MoE models.

    Inverse of :func:`qwen3_moe_hf_to_tune`: unstacks 3D expert tensors into
    individual per-expert weights and transposes back to ``[out_dim, in_dim]``.

    Args:
        state_dict (dict[str, torch.Tensor]): Torchtune state dict.
        num_heads (int): Number of attention heads.
        num_kv_heads (int): Number of KV heads.
        dim (int): Hidden dimension.
        head_dim (int): Head dimension.
        tie_word_embeddings (bool): Whether embeddings are tied.

    Returns:
        dict[str, torch.Tensor]: State dict in HF's format.
    """
    converted = {}
    inverted = {v: k for k, v in _FROM_HF_MOE.items() if v is not None}
    if head_dim is None:
        head_dim = dim // num_heads

    for key, value in state_dict.items():
        # Expert weights: layers.{L}.mlp.experts.{proj_name}
        if ".mlp.experts." in key and not any(
            lora_key in key for lora_key in ("lora_gate", "lora_up", "lora_down")
        ):
            parts = key.split(".")
            layer_idx = int(parts[1])
            proj_name = parts[4]  # gate_proj, up_proj, or down_proj

            num_experts = value.shape[0]
            for e in range(num_experts):
                # Transpose back from [in_dim, out_dim] to HF's [out_dim, in_dim]
                hf_key = f"model.layers.{layer_idx}.mlp.experts.{e}.{proj_name}.weight"
                converted[hf_key] = value[e].t()

        # Router gate: layers.{L}.mlp.router.gate.weight
        elif ".mlp.router.gate.weight" in key:
            layer_idx = int(key.split(".")[1])
            converted[f"model.layers.{layer_idx}.mlp.gate.weight"] = value

        # Standard keys
        else:
            new_key = get_mapped_key(key, inverted)
            converted[new_key] = value
            if QWEN3_TUNE_EMBEDDING_KEY in key and tie_word_embeddings:
                converted["lm_head.weight"] = value.detach().clone()

    return converted

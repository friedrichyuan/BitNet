"""BitNet b1.58 fake-quantization math shared by the trainer (HF) and rollout (vLLM) sides.

Both functions mirror ``transformers.integrations.bitnet.WeightQuant`` / ``ActQuant`` exactly
(fp32 compute, cast back to the input dtype), so the FSDP actor and the vLLM rollout see the
same ternary weights and int8 activations.
"""

import re

import torch

# Linear layers that BitNet replaces with AutoBitLinear. Embeddings, lm_head and norms stay high precision.
BITLINEAR_WEIGHT_RE = re.compile(r"\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.weight$")


def is_bitlinear_weight(name: str) -> bool:
    return BITLINEAR_WEIGHT_RE.search(name) is not None


def weight_quant(weight: torch.Tensor) -> torch.Tensor:
    """Per-tensor absmean ternarization: w -> round(w / mean|w|).clamp(-1, 1) * mean|w|."""
    dtype = weight.dtype
    weight = weight.float()
    scale = 1.0 / weight.abs().mean().clamp_(min=1e-5)
    weight = (weight * scale).round().clamp(-1, 1) / scale
    return weight.to(dtype)


def act_quant(activation: torch.Tensor) -> torch.Tensor:
    """Per-token absmax int8 fake-quantization over the last dim."""
    dtype = activation.dtype
    activation = activation.float()
    scale = 127 / activation.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    activation = (activation * scale).round().clamp(-128, 127) / scale
    return activation.to(dtype)


def ternary_codes(weight: torch.Tensor) -> torch.Tensor:
    """Ternary codes in {-1, 0, 1} (int8) that ``weight_quant`` would produce."""
    weight = weight.float()
    scale = 1.0 / weight.abs().mean().clamp_(min=1e-5)
    return (weight * scale).round().clamp(-1, 1).to(torch.int8)

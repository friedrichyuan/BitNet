"""transformers-side fixes for BitNet under verl.

1. Run transformers' BitNet QAT ops eagerly.

Upstream ``WeightQuant.forward`` / ``ActQuant.forward`` are wrapped in ``@torch.compile``. Under
verl (dynamic batch sizes, gradient checkpointing, FSDP2) that recompiles on most new shapes, and
compiled numerics can drift from the eager ops vLLM uses. ``AutoBitLinear.forward`` looks these
classes up as module globals at call time, so swapping the globals is enough.

Set ``AGENTICRL_BITNET_HF_COMPILE=1`` to keep the upstream compiled versions.

2. Drop the dangling ``auto_map`` from ``BitNetConfig``.

microsoft/bitnet-b1.58-2B-4T-bf16's config.json still maps to ``modeling_bitnet.py`` remote code that the
checkpoint does not ship; the model loads through transformers' native class instead. verl's FSDP
checkpoint manager calls ``custom_object_save`` whenever the config has ``auto_map``, which then tries to
copy transformers' own ``modeling_bitnet.py`` and its relative imports and dies with FileNotFoundError on
the first checkpoint save.
"""

import functools
import os

import torch

from .quant import act_quant, weight_quant


class WeightQuant(torch.autograd.Function):
    """Ternary weight quantization with a straight-through estimator."""

    @staticmethod
    def forward(ctx, weight):
        return weight_quant(weight)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()


class ActQuant(torch.autograd.Function):
    """Int8 per-token activation quantization with a straight-through estimator."""

    @staticmethod
    def forward(ctx, activation):
        return act_quant(activation)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()


# transformers' original (compiled) classes, kept for tests and debugging.
UPSTREAM: dict[str, type[torch.autograd.Function]] = {}


def drop_bitnet_auto_map() -> None:
    from transformers import BitNetConfig

    if getattr(BitNetConfig.__init__, "_agenticrl_drops_auto_map", False):
        return
    original_init = BitNetConfig.__init__

    @functools.wraps(original_init)
    def __init__(self, *args, **kwargs):
        kwargs.pop("auto_map", None)
        original_init(self, *args, **kwargs)

    __init__._agenticrl_drops_auto_map = True
    BitNetConfig.__init__ = __init__


def apply_hf_patch() -> bool:
    if os.environ.get("AGENTICRL_BITNET_HF_COMPILE", "0") == "1":
        return False
    from transformers.integrations import bitnet

    if bitnet.WeightQuant is not WeightQuant:
        UPSTREAM.update(WeightQuant=bitnet.WeightQuant, ActQuant=bitnet.ActQuant)
    bitnet.WeightQuant = WeightQuant
    bitnet.ActQuant = ActQuant
    return True

"""vLLM quantization method for BitNet b1.58 checkpoints with bf16 latent weights.

The HF config ships ``quantization_config = {"quant_method": "bitnet", "linear_class":
"autobitlinear", "quantization_mode": "online"}``, which vLLM does not know. This registers
``bitnet`` so every ``LinearBase`` gets ``BitNetLinearMethod``: int8 per-token activation
fake-quant, then a regular bf16 GEMM. Ternarizing the weights happens in
``BitNetForCausalLM.load_weights``, on each full checkpoint tensor before q/k/v and gate/up get
merged, because BitNet uses one absmean scale per original projection.
"""

from typing import Any

import torch
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig, QuantizeMethodBase

from .quant import act_quant


class BitNetLinearMethod(UnquantizedLinearMethod):
    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        return super().apply(layer, act_quant(x), bias)


@register_quantization_config("bitnet")
class BitNetQuantConfig(QuantizationConfig):
    def __init__(self) -> None:
        super().__init__()

    def __repr__(self) -> str:
        return "BitNetQuantConfig(linear_class=autobitlinear, quantization_mode=online)"

    def get_name(self) -> str:
        return "bitnet"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "BitNetQuantConfig":
        linear_class = config.get("linear_class", "bitlinear")
        quantization_mode = config.get("quantization_mode", "offline")
        if linear_class != "autobitlinear" or quantization_mode != "online":
            raise ValueError(
                "agenticrl's vLLM BitNet backend only supports bf16 latent-weight checkpoints "
                "(linear_class=autobitlinear, quantization_mode=online, e.g. "
                f"microsoft/bitnet-b1.58-2B-4T-bf16), got linear_class={linear_class}, "
                f"quantization_mode={quantization_mode}."
            )
        if config.get("use_rms_norm", False):
            raise ValueError("BitNet quantization_config.use_rms_norm=True is not supported.")
        return cls()

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> QuantizeMethodBase | None:
        if isinstance(layer, LinearBase):
            return BitNetLinearMethod()
        return None

"""vLLM implementation of ``BitNetForCausalLM`` (BitNet b1.58, bf16 latent weights).

Matches ``transformers.models.bitnet`` with ``AutoBitLinear`` in online mode:

* Llama layout plus ``attn_sub_norm`` (before ``o_proj``) and ``ffn_sub_norm`` (before ``down_proj``);
* MLP is ``down(ffn_sub_norm(relu(gate)^2 * up))``;
* every q/k/v/o/gate/up/down weight is ternarized per tensor when it is loaded (``load_weights``),
  and the input of every linear layer is int8 fake-quantized (``BitNetLinearMethod``).

Only TP=1 / PP=1 is supported: the sub-norms and the per-token activation absmax run over the full
hidden / intermediate dim, which tensor parallelism would shard.
"""

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import MergedColumnParallelLinear, RowParallelLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.llama import LlamaAttention, LlamaForCausalLM

from .quant import is_bitlinear_weight, weight_quant


class BitNetMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        rms_norm_eps: float,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if hidden_act != "relu2":
            raise ValueError(f"Unsupported activation: {hidden_act}. BitNet uses relu2.")
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.ffn_sub_norm = RMSNorm(intermediate_size, eps=rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        x = self.ffn_sub_norm(torch.square(F.relu(gate)) * up)
        x, _ = self.down_proj(x)
        return x


class BitNetAttention(LlamaAttention):
    def __init__(self, *, config, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        self.attn_sub_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(self.attn_sub_norm(attn_output))
        return output


class BitNetDecoderLayer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.hidden_size = config.hidden_size
        self.self_attn = BitNetAttention(
            config=config,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(config, "num_key_value_heads", config.num_attention_heads),
            max_position_embeddings=getattr(config, "max_position_embeddings", 4096),
            quant_config=quant_config,
            bias=getattr(config, "attention_bias", False),
            bias_o_proj=getattr(config, "attention_bias", False),
            cache_config=vllm_config.cache_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = BitNetMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            rms_norm_eps=config.rms_norm_eps,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class BitNetForCausalLM(LlamaForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        tp_size = get_tensor_model_parallel_world_size()
        if tp_size != 1 or get_pp_group().world_size != 1:
            raise ValueError(
                f"BitNetForCausalLM supports tensor_parallel_size=1 and pipeline_parallel_size=1 only "
                f"(got tp={tp_size}, pp={get_pp_group().world_size}). "
                "Set actor_rollout_ref.rollout.tensor_model_parallel_size=1."
            )
        super().__init__(vllm_config=vllm_config, prefix=prefix, layer_type=BitNetDecoderLayer)
        # The trainer ternarizes the bf16-cast params (FSDP mixed precision), so cast to the rollout
        # dtype first: verl may stream fp32 master weights.
        self._weight_dtype = vllm_config.model_config.dtype

    def _ternarize(self, weights: Iterable[tuple[str, torch.Tensor]]) -> Iterable[tuple[str, torch.Tensor]]:
        for name, tensor in weights:
            if is_bitlinear_weight(name):
                tensor = weight_quant(tensor.to(self._weight_dtype))
            yield name, tensor

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return super().load_weights(self._ternarize(weights))

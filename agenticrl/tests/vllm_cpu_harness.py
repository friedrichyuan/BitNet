"""CPU harness for agenticrl's vLLM BitNet model; prints one JSON line. Run by test_bitnet_vllm.py.

vLLM's paged-attention kernel needs a GPU, so each layer's ``self_attn.attn`` is replaced by causal
SDPA over one sequence, and vLLM's CPU GEMM dispatch (which needs a CPU-specific weight prep step)
by ``F.linear`` -- what the CUDA unquantized path computes. Everything else is the real code path:
weight loading + ternarization, BitNetLinearMethod, sub-norms, relu2, rotary, RMSNorm, tied lm_head.

    python -m agenticrl.tests.vllm_cpu_harness {float32,bfloat16}
"""

import json
import sys
import tempfile

import torch
import torch.nn.functional as F
import vllm.platforms

from vllm.platforms.cpu import CpuPlatform  # isort: skip

vllm.platforms._current_platform = CpuPlatform()  # must precede any other vllm import

from safetensors.torch import load_file  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402
from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config  # noqa: E402
from vllm.distributed import init_distributed_environment, initialize_model_parallel  # noqa: E402
from vllm.model_executor.layers import linear as vllm_linear  # noqa: E402

import agenticrl.bitnet  # noqa: E402,F401
from agenticrl.bitnet.quant import weight_quant  # noqa: E402
from agenticrl.tests.tiny import make_tiny_bitnet  # noqa: E402

vllm_linear.dispatch_unquantized_gemm = lambda: (lambda layer, x, w, b=None: F.linear(x, w, b))


class CausalSDPA(torch.nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.h, self.kvh, self.d, self.scale = attn.num_heads, attn.num_kv_heads, attn.head_dim, attn.scaling

    def forward(self, q, k, v):
        t = q.shape[0]
        q = q.view(t, self.h, self.d).transpose(0, 1)
        k = k.view(t, self.kvh, self.d).transpose(0, 1).repeat_interleave(self.h // self.kvh, 0)
        v = v.view(t, self.kvh, self.d).transpose(0, 1).repeat_interleave(self.h // self.kvh, 0)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scale)
        return o.transpose(0, 1).reshape(t, self.h * self.d)


def vllm_logits(model, ids):
    pos = torch.arange(ids.numel())
    h, res = model.model.embed_input_ids(ids), None
    for layer in model.model.layers:
        h, res = layer(pos, h, res)
    h, _ = model.model.norm(h, res)
    return F.linear(h, model.lm_head.weight).float()


def main(dtype_name: str):
    dtype = getattr(torch, dtype_name)
    path = tempfile.mkdtemp()
    make_tiny_bitnet(path, dtype=torch.bfloat16)
    sd = load_file(f"{path}/model.safetensors")
    hf = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, attn_implementation="sdpa").eval()

    vllm_config = VllmConfig(model_config=ModelConfig(model=path, dtype=dtype, skip_tokenizer_init=True))
    out = {"quant_config": repr(vllm_config.quant_config)}
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=1, rank=0, local_rank=0, distributed_init_method=f"file://{path}/pg", backend="gloo"
        )
        initialize_model_parallel(1, 1)
        from agenticrl.bitnet.vllm_model import BitNetForCausalLM

        torch.set_default_dtype(dtype)
        model = BitNetForCausalLM(vllm_config=vllm_config).eval()
        loaded = model.load_weights(list(sd.items()))

        layer = model.model.layers[1]
        w = {k[len("model.layers.1.") :]: weight_quant(v.to(dtype)) for k, v in sd.items() if k.startswith("model.layers.1.")}
        out["linear_method"] = type(layer.mlp.gate_up_proj.quant_method).__name__
        out["unloaded_params"] = sorted({n for n, _ in model.named_parameters()} - loaded)
        out["qkv_exact"] = torch.equal(
            layer.self_attn.qkv_proj.weight,
            torch.cat([w[f"self_attn.{p}_proj.weight"] for p in "qkv"]),
        )
        out["gate_up_exact"] = torch.equal(
            layer.mlp.gate_up_proj.weight, torch.cat([w["mlp.gate_proj.weight"], w["mlp.up_proj.weight"]])
        )
        out["down_o_exact"] = torch.equal(layer.mlp.down_proj.weight, w["mlp.down_proj.weight"]) and torch.equal(
            layer.self_attn.o_proj.weight, w["self_attn.o_proj.weight"]
        )
        out["lm_head_tied"] = model.lm_head.weight is model.model.embed_tokens.weight
        before = layer.self_attn.qkv_proj.weight.clone()
        model.load_weights([(k, v.float()) for k, v in sd.items()])  # verl may stream fp32 master weights
        out["fp32_stream_same"] = torch.equal(layer.self_attn.qkv_proj.weight, before)

        for lyr in model.model.layers:
            lyr.self_attn.attn = CausalSDPA(lyr.self_attn)
        torch.manual_seed(1)
        ids = torch.randint(0, 512, (48,))
        with torch.no_grad():
            hf_lp = hf(ids[None]).logits[0].float().log_softmax(-1)
            v_lp = vllm_logits(model, ids).log_softmax(-1)
            # Control: the same model without activation quantization must disagree with HF.
            for lyr in model.model.layers:
                for mod in (lyr.self_attn.qkv_proj, lyr.self_attn.o_proj, lyr.mlp.gate_up_proj, lyr.mlp.down_proj):
                    mod.quant_method = vllm_linear.UnquantizedLinearMethod()
            c_lp = vllm_logits(model, ids).log_softmax(-1)

    kl = lambda a, b: F.kl_div(a, b, log_target=True, reduction="none").sum(-1)  # noqa: E731  KL(b || a)
    tok = ids[1:, None]
    out["token_logprob_abs_diff_max"] = (hf_lp[:-1].gather(-1, tok) - v_lp[:-1].gather(-1, tok)).abs().max().item()
    out["argmax_agreement"] = (hf_lp.argmax(-1) == v_lp.argmax(-1)).float().mean().item()
    out["kl_mean"] = kl(v_lp, hf_lp).mean().item()
    out["control_kl_mean"] = kl(c_lp, hf_lp).mean().item()
    print("HARNESS_RESULT " + json.dumps(out))


if __name__ == "__main__":
    main(sys.argv[1])

"""transformers side: the model verl's FSDP actor trains."""

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.integrations import bitnet as hf_bitnet

import agenticrl.bitnet  # noqa: F401
from agenticrl.bitnet import hf_patch
from agenticrl.bitnet.quant import act_quant, ternary_codes, weight_quant
from agenticrl.tests.tiny import make_tiny_bitnet

PROJ = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("tiny_bitnet"))
    make_tiny_bitnet(path)
    return path


def test_quant_math():
    w = torch.randn(64, 32) * 2
    q = weight_quant(w)
    scale = w.abs().mean()
    assert torch.allclose(q.unique().abs().max(), scale, rtol=1e-5)
    assert set(ternary_codes(w).unique().tolist()) <= {-1, 0, 1}
    torch.testing.assert_close(q, ternary_codes(w).float() * scale, rtol=1e-6, atol=1e-6)
    x = torch.randn(5, 7, 32)
    xq = act_quant(x)
    step = x.abs().amax(-1, keepdim=True) / 127
    assert ((xq / step) - (xq / step).round()).abs().max() < 1e-3  # on the int8 grid
    assert (xq - x).abs().le(step / 2 + 1e-6).all()


def test_patch_installed():
    assert hf_bitnet.WeightQuant is hf_patch.WeightQuant
    assert hf_bitnet.ActQuant is hf_patch.ActQuant


def test_auto_map_dropped(tiny_dir):
    cfg = AutoConfig.from_pretrained(tiny_dir)
    cfg_with_map = type(cfg).from_dict({**cfg.to_dict(), "auto_map": {"AutoConfig": "configuration_bitnet.BitNetConfig"}})
    assert not hasattr(cfg_with_map, "auto_map")
    assert cfg_with_map.quantization_config["quant_method"] == "bitnet"


def test_from_pretrained_is_online_qat(tiny_dir):
    model = AutoModelForCausalLM.from_pretrained(tiny_dir, dtype=torch.float32)
    for layer in model.model.layers:
        for mod in (*(getattr(layer.self_attn, p, None) for p in PROJ[:4]), *(getattr(layer.mlp, p) for p in PROJ[4:])):
            assert isinstance(mod, hf_bitnet.AutoBitLinear) and mod.online_quant
    assert type(model.lm_head) is torch.nn.Linear
    assert model.lm_head.weight is model.model.embed_tokens.weight


def test_eager_patch_matches_upstream(tiny_dir):
    upstream = hf_patch.UPSTREAM
    assert upstream, "upstream classes not recorded"
    model = AutoModelForCausalLM.from_pretrained(tiny_dir, dtype=torch.bfloat16).eval()
    ids = torch.randint(0, 512, (2, 24))
    with torch.no_grad():
        patched = model(ids).logits
        hf_bitnet.WeightQuant, hf_bitnet.ActQuant = upstream["WeightQuant"], upstream["ActQuant"]
        try:
            original = model(ids).logits
        except Exception as e:  # torch.compile needs a working C++ toolchain on CPU
            pytest.skip(f"upstream compiled ops unavailable here: {type(e).__name__}")
        finally:
            hf_bitnet.WeightQuant, hf_bitnet.ActQuant = hf_patch.WeightQuant, hf_patch.ActQuant
    torch.testing.assert_close(patched, original, rtol=0, atol=0)


def test_ste_gradients_reach_latent_weights(tiny_dir):
    model = AutoModelForCausalLM.from_pretrained(tiny_dir, dtype=torch.float32)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    ids = torch.randint(0, 512, (2, 16))
    model(ids, labels=ids).loss.backward()
    q = model.model.layers[0].self_attn.q_proj.weight
    assert q.grad is not None and q.grad.abs().sum() > 0
    assert model.model.layers[0].mlp.ffn_sub_norm.weight.grad.abs().sum() > 0

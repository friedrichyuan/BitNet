import os
import shutil

import torch
from safetensors.torch import load_file, save_file

from agenticrl.bitnet.quant import ternary_codes
from agenticrl.tests.tiny import make_tiny_bitnet
from agenticrl.tools.ternary_drift import compare

Q = "model.layers.0.self_attn.q_proj.weight"


def test_counts_planted_flips(tmp_path):
    base, target = str(tmp_path / "base"), str(tmp_path / "target")
    make_tiny_bitnet(base)
    shutil.copytree(base, target)
    sd = load_file(os.path.join(target, "model.safetensors"))

    # Plant 7 sign flips in q_proj (a large-magnitude +-1 weight negated keeps the absmean scale).
    q = sd[Q].clone()
    codes = ternary_codes(q).flatten()
    idx = (codes != 0).nonzero()[:7, 0]
    q.view(-1)[idx] = -q.view(-1)[idx]
    sd[Q] = q
    # And nudge a norm weight, which is not ternarized.
    sd["model.norm.weight"] = sd["model.norm.weight"] * 1.01
    save_file({k: v.contiguous() for k, v in sd.items()}, os.path.join(target, "model.safetensors"), metadata={"format": "pt"})

    flips, other, _ = compare(base, target)
    assert flips["q_proj"]["changed"] == 7 and flips["q_proj"]["sign_flips"] == 7
    assert all(c["changed"] == 0 for g, c in flips.items() if g != "q_proj")
    torch.testing.assert_close(torch.tensor(other["model.norm.weight"]), torch.tensor(0.01), rtol=0.05, atol=0)
    assert other["model.embed_tokens.weight"] == 0.0


def test_identical_checkpoints(tmp_path):
    base = str(tmp_path / "base")
    make_tiny_bitnet(base)
    flips, other, _ = compare(base, base)
    assert sum(c["changed"] for c in flips.values()) == 0
    assert max(other.values()) == 0.0

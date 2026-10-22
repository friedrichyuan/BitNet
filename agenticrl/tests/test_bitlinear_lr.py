"""Separate AdamW learning rate for BitLinear latent weights (agenticrl.bitnet.verl_patch, part 2)."""

import json
import os
import subprocess
import sys
import tempfile
import textwrap
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("verl")

import verl.workers.engine.fsdp.transformer_impl as transformer_impl  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402
from verl.workers.config.optimizer import FSDPOptimizerConfig  # noqa: E402

from agenticrl.bitnet.verl_patch import apply_verl_patch, bitlinear_param_groups  # noqa: E402
from agenticrl.tests.tiny import make_tiny_bitnet  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
apply_verl_patch()  # transformer_impl is already imported here, so this patches it directly


@pytest.fixture()
def model(tmp_path):
    make_tiny_bitnet(str(tmp_path), dtype=torch.float32)
    return AutoModelForCausalLM.from_pretrained(str(tmp_path), dtype=torch.float32)


def build(model, **optim):
    engine = SimpleNamespace(optimizer_config=FSDPOptimizerConfig(**optim))
    return transformer_impl.FSDPEngine._build_optimizer(engine, model)


def test_param_groups(model):
    base, bitlinear = bitlinear_param_groups(model.named_parameters(), 1e-5, 1e-4)
    # 3 layers x 7 projections; embed (lm_head tied) + 3 x 4 norms + final norm
    assert len(bitlinear["params"]) == 21 and len(base["params"]) == 14
    assert bitlinear["lr"] == 1e-4 and base["lr"] == 1e-5
    assert all(p.ndim == 2 for p in bitlinear["params"])


def test_flat_params_rejected():
    flat = [("_fsdp_wrapped_module._flat_param", torch.nn.Parameter(torch.zeros(4)))]
    with pytest.raises(ValueError, match="strategy=fsdp2"):
        bitlinear_param_groups(flat, 1e-5, 1e-4)


def test_default_is_one_group(model):
    optimizer = build(model, lr=1e-5)
    assert len(optimizer.param_groups) == 1 and optimizer.param_groups[0]["lr"] == 1e-5


def test_bitlinear_lr_moves_weights_by_their_own_lr(model):
    optimizer = build(model, lr=1e-5, weight_decay=0.0, override_optimizer_config={"bitlinear_lr": 1e-4})
    assert type(optimizer).__name__ == "AdamW"
    assert [g["lr"] for g in optimizer.param_groups] == [1e-5, 1e-4]  # verl logs actor/lr from group 0
    assert "bitlinear_lr" not in optimizer.defaults

    q = model.model.layers[0].self_attn.q_proj.weight
    norm = model.model.layers[0].input_layernorm.weight
    q0, norm0 = q.detach().clone(), norm.detach().clone()
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    optimizer.step()  # Adam's first step moves every weight by ~lr
    torch.testing.assert_close((q0 - q.detach()).abs().mean(), torch.tensor(1e-4), rtol=1e-2, atol=0)
    torch.testing.assert_close((norm0 - norm.detach()).abs().mean(), torch.tensor(1e-5), rtol=1e-2, atol=0)  # fp32 rounding near |w|~1


def test_lr_scheduler_keeps_the_ratio(model):
    from verl.utils.torch_functional import get_constant_schedule_with_warmup

    optimizer = build(model, lr=1e-5, override_optimizer_config={"bitlinear_lr": 1e-4})
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=4)
    for _ in range(2):
        optimizer.step()
        scheduler.step()
    base_lr, bitlinear_lr = scheduler.get_last_lr()
    assert 0 < base_lr < 1e-5 and bitlinear_lr == pytest.approx(10 * base_lr)


def test_names_survive_fsdp2(model):
    from torch.distributed.fsdp import fully_shard

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            "gloo", init_method=f"file://{tempfile.mkdtemp()}/pg", rank=0, world_size=1
        )
    try:
        for layer in model.model.layers:
            fully_shard(layer)
        fully_shard(model)
    except Exception as e:  # pragma: no cover - depends on the torch build
        pytest.skip(f"FSDP2 unavailable on CPU here: {type(e).__name__}: {e}")
    base, bitlinear = bitlinear_param_groups(model.named_parameters(), 1e-5, 1e-4)
    assert len(bitlinear["params"]) == 21 and len(base["params"]) == 14
    assert all(type(p).__name__ == "DTensor" for p in bitlinear["params"])


CHECK = textwrap.dedent(
    """
    import json
    {imports}
    import verl.workers.engine.fsdp.transformer_impl as t
    print("RESULT " + json.dumps(getattr(t.FSDPEngine._build_optimizer, "_agenticrl_bitnet", False)))
    """
)


@pytest.mark.parametrize(
    "imports, env",
    [
        ("import verl", {"VERL_USE_EXTERNAL_MODULES": "agenticrl.bitnet"}),  # every Ray worker
        ("import verl.workers.engine.fsdp.transformer_impl\nimport agenticrl.bitnet", {}),  # external_lib
    ],
)
def test_patch_applies_in_both_import_orders(imports, env):
    full_env = {**os.environ, "PYTHONPATH": ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")}
    full_env.pop("VERL_USE_EXTERNAL_MODULES", None)
    full_env.update(env)
    proc = subprocess.run(
        [sys.executable, "-c", CHECK.format(imports=imports)],
        capture_output=True, text=True, env=full_env, cwd=ROOT, timeout=600,
    )
    lines = [line for line in proc.stdout.splitlines() if line.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
    assert json.loads(lines[-1][len("RESULT ") :]) is True

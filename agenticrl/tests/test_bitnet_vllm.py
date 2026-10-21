"""vLLM side: agenticrl's BitNetForCausalLM must match transformers (the actor) token for token."""

import json
import os
import subprocess
import sys

import pytest

pytest.importorskip("vllm")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def run_harness(dtype: str) -> dict:
    env = {**os.environ, "PYTHONPATH": ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")}
    proc = subprocess.run(
        [sys.executable, "-m", "agenticrl.tests.vllm_cpu_harness", dtype],
        capture_output=True, text=True, env=env, cwd=ROOT, timeout=900,
    )
    lines = [line for line in proc.stdout.splitlines() if line.startswith("HARNESS_RESULT ")]
    assert proc.returncode == 0 and lines, f"harness failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    return json.loads(lines[-1][len("HARNESS_RESULT ") :])


@pytest.fixture(scope="module")
def fp32():
    return run_harness("float32")


def test_registration_and_weight_loading(fp32):
    assert fp32["quant_config"].startswith("BitNetQuantConfig")
    assert fp32["linear_method"] == "BitNetLinearMethod"
    assert fp32["unloaded_params"] == []
    # q/k/v and gate/up are ternarized with their own scale *before* vLLM merges them.
    assert fp32["qkv_exact"] and fp32["gate_up_exact"] and fp32["down_o_exact"]
    assert fp32["lm_head_tied"]
    assert fp32["fp32_stream_same"]


def test_fp32_logits_match_hf_exactly(fp32):
    assert fp32["argmax_agreement"] == 1.0
    assert fp32["token_logprob_abs_diff_max"] < 1e-4
    assert fp32["kl_mean"] < 1e-6
    assert fp32["control_kl_mean"] > 1e-3  # the comparison is sensitive to the BitNet-specific ops


def test_bf16_close_to_hf():
    # bf16 differs only by vLLM vs HF rounding order (RMSNorm / residual), amplified by int8 rounding
    # in a random network; the real-model number comes from scripts/check_rollout_parity.py on GPU.
    bf16 = run_harness("bfloat16")
    assert bf16["kl_mean"] < 1e-2
    assert bf16["kl_mean"] < bf16["control_kl_mean"]

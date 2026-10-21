"""verl must not look for a trainer-side quantized weight exporter for BitNet (it would raise in init_model)."""

import json
import os
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("verl")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

CHECK = textwrap.dedent(
    """
    import json, sys, tempfile
    {imports}
    from hydra import compose, initialize_config_dir
    from transformers import AutoConfig

    from agenticrl.tests.tiny import make_tiny_bitnet

    with initialize_config_dir(config_dir="{root}/agenticrl/config", version_base=None):
        rollout_config = compose(config_name="grpo_gsm8k_bitnet").actor_rollout_ref.rollout

    class ModelConfig:  # the two attributes dispatch_rollout_weight_quantizer reads
        def __init__(self, hf_config):
            self.hf_config, self.local_path = hf_config, None

    path = tempfile.mkdtemp()
    make_tiny_bitnet(path)
    bitnet = ModelConfig(AutoConfig.from_pretrained(path))
    awq = ModelConfig(AutoConfig.from_pretrained(path))
    awq.hf_config.quantization_config = {{"quant_method": "awq"}}

    dispatch = engine_workers.dispatch_rollout_weight_quantizer
    out = {{"patched": getattr(dispatch, "_agenticrl_bitnet", False)}}
    out["bitnet"] = repr(dispatch(model_config=bitnet, rollout_config=rollout_config, qat_config=None))
    try:
        dispatch(model_config=awq, rollout_config=rollout_config, qat_config=None)
        out["awq"] = "no error"
    except NotImplementedError as e:
        out["awq"] = "NotImplementedError"
    print("RESULT " + json.dumps(out))
    """
)


def run(imports: str, extra_env: dict) -> dict:
    env = {**os.environ, "PYTHONPATH": ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")}
    env.pop("VERL_USE_EXTERNAL_MODULES", None)
    env.update(extra_env)
    proc = subprocess.run(
        [sys.executable, "-c", CHECK.format(imports=imports, root=ROOT)],
        capture_output=True, text=True, env=env, cwd=ROOT, timeout=600,
    )
    lines = [line for line in proc.stdout.splitlines() if line.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, f"check failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    return json.loads(lines[-1][len("RESULT ") :])


def expected(result):
    assert result["patched"]
    assert result["bitnet"] == "None"  # no trainer-side exporter: vLLM ternarizes bf16 weights itself
    assert result["awq"] == "NotImplementedError"  # every other quant_method still goes to verl


def test_env_var_import_order():
    # What every Ray worker does: `import verl` imports agenticrl.bitnet before engine_workers exists.
    expected(run("import verl.workers.engine_workers as engine_workers", {"VERL_USE_EXTERNAL_MODULES": "agenticrl.bitnet"}))


def test_external_lib_import_order():
    # engine_workers already bound the original function; agenticrl.bitnet is imported afterwards.
    expected(run("import verl.workers.engine_workers as engine_workers\nimport agenticrl.bitnet", {}))

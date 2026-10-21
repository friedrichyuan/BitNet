"""verl-side fix: no trainer-side quantized weight export for BitNet.

verl sees ``quantization_config`` in the HF config and assumes vLLM was started on an
already-quantized checkpoint, so the trainer would have to quantize every weight refit itself
(``dispatch_rollout_weight_quantizer``). It only knows compressed-tensors and FP8 and raises
NotImplementedError for ``quant_method="bitnet"`` in ``init_model``.

For BitNet the trainer streams the bf16 latent weights unchanged and
``BitNetForCausalLM.load_weights`` on the vLLM side ternarizes them, so no exporter is needed:
return None for BitNet and leave every other case (including an explicit ``rollout.quantization``
override or verl QAT) to verl.

``engine_workers`` binds the function with ``from verl.utils.qat import ...`` at import time, so
the patch has to land on ``verl.utils.qat`` before that import. ``VERL_USE_EXTERNAL_MODULES``
makes ``verl/__init__.py`` import this package first in every process, which guarantees that; an
already-imported ``engine_workers`` is patched as well, for the ``external_lib`` import path.
"""

import functools
import sys


def _native_quant_method(model_config) -> str | None:
    hf_config = getattr(model_config, "hf_config", None)
    native = getattr(hf_config, "quantization_config", None)
    if native is None:
        native = getattr(getattr(hf_config, "text_config", None), "quantization_config", None)
    if hasattr(native, "to_dict"):
        native = native.to_dict()
    if not isinstance(native, dict):
        return None
    return str(native.get("quant_method", "")).lower()


def _has_rollout_quantization_override(rollout_config) -> bool:
    quantization = getattr(rollout_config, "quantization", None)
    return any(getattr(quantization, key, None) for key in ("linear", "moe", "config_file"))


def apply_verl_patch() -> bool:
    try:
        import verl.utils.qat as verl_qat
        from verl.utils.qat import compressed_tensors
    except ImportError:  # verl not importable here (e.g. a plain vLLM / transformers process)
        return False

    original = compressed_tensors.dispatch_rollout_weight_quantizer
    if getattr(original, "_agenticrl_bitnet", False):
        return True

    @functools.wraps(original)
    def dispatch_rollout_weight_quantizer(*, model_config, rollout_config, qat_config=None):
        qat_enabled = qat_config is not None and getattr(qat_config, "enable", False)
        if (
            not qat_enabled
            and not _has_rollout_quantization_override(rollout_config)
            and _native_quant_method(model_config) == "bitnet"
        ):
            return None
        return original(model_config=model_config, rollout_config=rollout_config, qat_config=qat_config)

    dispatch_rollout_weight_quantizer._agenticrl_bitnet = True
    compressed_tensors.dispatch_rollout_weight_quantizer = dispatch_rollout_weight_quantizer
    verl_qat.dispatch_rollout_weight_quantizer = dispatch_rollout_weight_quantizer
    engine_workers = sys.modules.get("verl.workers.engine_workers")
    if engine_workers is not None:
        engine_workers.dispatch_rollout_weight_quantizer = dispatch_rollout_weight_quantizer
    return True

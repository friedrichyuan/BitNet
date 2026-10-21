"""BitNet b1.58 support for verl.

Importing this package
* swaps transformers' ``@torch.compile``'d BitNet QAT ops for eager ones and drops the dangling
  ``auto_map`` from ``BitNetConfig`` (see ``hf_patch``), and
* registers the ``bitnet`` quantization method and ``BitNetForCausalLM`` with vLLM.

verl imports it through ``actor_rollout_ref.model.external_lib=agenticrl.bitnet`` and through
``VERL_USE_EXTERNAL_MODULES=agenticrl.bitnet``. The env var matters for the vLLM server actor and
its spawned worker processes, which never build an ``HFModelConfig`` themselves but all import verl.
"""

from .hf_patch import apply_hf_patch, drop_bitnet_auto_map

VLLM_ARCH = "BitNetForCausalLM"


def register_vllm() -> bool:
    try:
        from vllm import ModelRegistry
    except ImportError:
        return False

    from . import vllm_quant  # noqa: F401  (registers the "bitnet" quantization method)

    if VLLM_ARCH not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(VLLM_ARCH, "agenticrl.bitnet.vllm_model:BitNetForCausalLM")
    return True


apply_hf_patch()
drop_bitnet_auto_map()
register_vllm()

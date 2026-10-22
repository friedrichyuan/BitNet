"""verl-side fixes for BitNet.

1. No trainer-side quantized weight export.

   verl sees ``quantization_config`` in the HF config and assumes vLLM was started on an
   already-quantized checkpoint, so the trainer would have to quantize every weight refit itself
   (``dispatch_rollout_weight_quantizer``). It only knows compressed-tensors and FP8 and raises
   NotImplementedError for ``quant_method="bitnet"`` in ``init_model``. For BitNet the trainer streams
   the bf16 latent weights unchanged and ``BitNetForCausalLM.load_weights`` on the vLLM side ternarizes
   them, so no exporter is needed: return None for BitNet and leave every other case (an explicit
   ``rollout.quantization`` override, verl QAT) to verl.

   ``engine_workers`` binds the function with ``from verl.utils.qat import ...`` at import time, so the
   patch has to land on ``verl.utils.qat`` before that import. ``VERL_USE_EXTERNAL_MODULES`` makes
   ``verl/__init__.py`` import this package first in every process, which guarantees that; an
   already-imported ``engine_workers`` is patched as well, for the ``external_lib`` import path.

2. A separate learning rate for the BitLinear latent weights.

   The FSDP forward ternarizes the bf16-cast weights, and near the ternary threshold bf16 values are
   0.004-0.008 apart, so a latent weight must move ~0.002 before its ternary code can flip. At a
   learning rate that is safe for RMSNorm weights and embeddings (1e-5) one epoch moves a weight at
   most ~1e-3: measured, 0 of 2.08B codes flipped. With

       actor_rollout_ref.actor.optim.override_optimizer_config={bitlinear_lr: 1e-4}

   the q/k/v/o/gate/up/down weights get their own AdamW param group at ``bitlinear_lr`` while every
   other parameter stays at ``optim.lr``. The key is consumed here and never reaches the optimizer.
   Param groups need per-parameter tensors, i.e. ``actor.strategy=fsdp2`` (FSDP1's default flat
   parameters mix every weight of a layer). The base group comes first: verl logs ``actor/lr`` from
   group 0 and the LR scheduler scales both groups by the same factor.

   ``FSDPEngine`` is imported long after this package (``verl/__init__`` imports it first), so the
   patch is applied by a post-import hook on ``verl.workers.engine.fsdp.transformer_impl``.
"""

import dataclasses
import functools
import importlib.abc
import importlib.util
import logging
import sys

from .quant import is_bitlinear_weight

logger = logging.getLogger(__name__)

BITLINEAR_LR_KEY = "bitlinear_lr"
FSDP_ENGINE_MODULE = "verl.workers.engine.fsdp.transformer_impl"


# ---------------------------------------------------------------------------------------------------
# 1. rollout weight export
# ---------------------------------------------------------------------------------------------------


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


def _patch_rollout_weight_quantizer() -> None:
    import verl.utils.qat as verl_qat
    from verl.utils.qat import compressed_tensors

    original = compressed_tensors.dispatch_rollout_weight_quantizer
    if getattr(original, "_agenticrl_bitnet", False):
        return

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


# ---------------------------------------------------------------------------------------------------
# 2. BitLinear learning rate
# ---------------------------------------------------------------------------------------------------


def bitlinear_param_groups(named_parameters, base_lr: float, bitlinear_lr: float) -> list[dict]:
    """[{base params, lr=base_lr}, {BitLinear latent weights, lr=bitlinear_lr}] over trainable params."""
    base, bitlinear = [], []
    for name, param in named_parameters:
        if not param.requires_grad:
            continue
        if "flat_param" in name:
            raise ValueError(
                f"optim.override_optimizer_config.{BITLINEAR_LR_KEY} needs per-parameter tensors, but the "
                f"optimizer sees FSDP1 flat parameters ({name}). Set actor_rollout_ref.actor.strategy=fsdp2."
            )
        (bitlinear if is_bitlinear_weight(name) else base).append(param)
    if not bitlinear:
        raise ValueError(f"{BITLINEAR_LR_KEY} is set but no q/k/v/o/gate/up/down_proj weights were found")
    return [{"params": base, "lr": base_lr}, {"params": bitlinear, "lr": bitlinear_lr}]


def _patch_fsdp_engine(module) -> None:
    engine_cls = module.FSDPEngine
    original = engine_cls._build_optimizer
    if getattr(original, "_agenticrl_bitnet", False):
        return

    @functools.wraps(original)
    def _build_optimizer(self, model):
        config = self.optimizer_config
        override = dict(config.override_optimizer_config or {})
        if override.get(BITLINEAR_LR_KEY) is None:
            return original(self, model)
        from verl.workers.config.optimizer import build_optimizer

        bitlinear_lr = float(override.pop(BITLINEAR_LR_KEY))
        groups = bitlinear_param_groups(model.named_parameters(), config.lr, bitlinear_lr)
        print(
            f"[agenticrl.bitnet] optimizer param groups: {len(groups[0]['params'])} tensors at lr={config.lr}, "
            f"{len(groups[1]['params'])} BitLinear weights at lr={bitlinear_lr}"
        )
        return build_optimizer(groups, dataclasses.replace(config, override_optimizer_config=override or None))

    _build_optimizer._agenticrl_bitnet = True
    engine_cls._build_optimizer = _build_optimizer


class _PatchAfterImport(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Run ``patch(module)`` right after ``fullname`` is first imported."""

    def __init__(self, fullname, patch):
        self.fullname, self.patch, self.spec = fullname, patch, None

    def find_spec(self, fullname, path, target=None):
        if fullname != self.fullname or self.spec is not None:
            return None
        sys.meta_path.remove(self)  # one-shot; also keeps the lookup below from recursing into us
        spec = importlib.util.find_spec(fullname)
        if spec is None or spec.loader is None:
            return spec
        self.spec, self.inner = spec, spec.loader
        spec.loader = self
        return spec

    def create_module(self, spec):
        return self.inner.create_module(spec)

    def exec_module(self, module):
        self.inner.exec_module(module)
        self.patch(module)


def _patch_when_imported(fullname, patch) -> None:
    if fullname in sys.modules:
        patch(sys.modules[fullname])
    elif not any(isinstance(f, _PatchAfterImport) and f.fullname == fullname for f in sys.meta_path):
        sys.meta_path.insert(0, _PatchAfterImport(fullname, patch))


def apply_verl_patch() -> bool:
    try:
        import verl.utils.qat  # noqa: F401
    except ImportError:  # verl not importable here (e.g. a plain vLLM / transformers process)
        return False
    _patch_rollout_weight_quantizer()
    _patch_when_imported(FSDP_ENGINE_MODULE, _patch_fsdp_engine)
    return True

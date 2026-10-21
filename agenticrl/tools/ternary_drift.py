"""How much did training actually change a BitNet model?

The actor trains bf16-cast fp32 *latent* weights, but the forward pass only ever sees their ternary
codes. If the learning rate is too small no code flips, and the policy can only move through the
few full-precision tensors (embeddings, RMSNorm weights). This compares a trained checkpoint against
the base model:

* q/k/v/o/gate/up/down: fraction of ternary codes that changed (split into 0<->+-1 and sign flips),
  and the relative change of the per-tensor absmean scale;
* everything else: relative L2 change ||w - w0|| / ||w0||.

Usage:
    python -m agenticrl.tools.ternary_drift \
        --base models/bitnet-b1.58-2B-4T-bf16 \
        --target checkpoints/agenticrl/<exp>/global_step_<N>/actor/huggingface
(the ``huggingface`` dir is written when ``actor.checkpoint.save_contents`` includes ``hf_model``).
"""

import argparse
import collections
import json
import os
import re

import torch
from safetensors import safe_open

from agenticrl.bitnet.quant import is_bitlinear_weight, ternary_codes


def tensor_index(model_dir: str) -> dict[str, str]:
    """Map tensor name -> safetensors file, for single-file and sharded checkpoints."""
    index = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index) as f:
            weight_map = json.load(f)["weight_map"]
        return {name: os.path.join(model_dir, file) for name, file in weight_map.items()}
    single = os.path.join(model_dir, "model.safetensors")
    with safe_open(single, framework="pt") as f:
        return {name: single for name in f.keys()}


def load(path: str, name: str) -> torch.Tensor:
    with safe_open(path, framework="pt") as f:
        return f.get_tensor(name)


def group_of(name: str) -> str:
    """'model.layers.12.self_attn.q_proj.weight' -> 'q_proj'; non-layer tensors keep their name."""
    m = re.match(r"model\.layers\.\d+\.(.+)\.weight$", name)
    return m.group(1).split(".")[-1] if m else name


def compare(base_dir: str, target_dir: str):
    """Return (per-group ternary flip counters, per-tensor rel. L2 change of other tensors, per-tensor stats)."""
    base, target = tensor_index(base_dir), tensor_index(target_dir)
    missing = sorted(set(base) - set(target) - {"lm_head.weight"})
    assert not missing, f"tensors missing from target: {missing[:5]}"

    flips = collections.defaultdict(collections.Counter)
    other, per_tensor = {}, {}
    for name in sorted(base):
        if name not in target:
            continue
        # The trainer ternarizes the bf16-cast params, so compare in bf16 as well.
        w0 = load(base[name], name).to(torch.bfloat16)
        w1 = load(target[name], name).to(torch.bfloat16)
        if is_bitlinear_weight(name):
            c0, c1 = ternary_codes(w0), ternary_codes(w1)
            changed = c0 != c1
            stats = {
                "n": c0.numel(),
                "changed": int(changed.sum()),
                "sign_flips": int((changed & (c0 != 0) & (c1 != 0)).sum()),
                "zero_to_nonzero": int((changed & (c0 == 0)).sum()),
                "nonzero_to_zero": int((changed & (c1 == 0)).sum()),
            }
            s0, s1 = w0.float().abs().mean().item(), w1.float().abs().mean().item()
            per_tensor[name] = {**stats, "scale_rel_change": (s1 - s0) / s0}
            flips[group_of(name)].update(stats)
        else:
            rel = ((w1.float() - w0.float()).norm() / w0.float().norm().clamp(min=1e-12)).item()
            other[name] = rel
            per_tensor[name] = {"rel_l2_change": rel}
    return flips, other, per_tensor


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--json", default=None, help="optional output file with per-tensor stats")
    args = parser.parse_args()

    flips, other, per_tensor = compare(args.base, args.target)

    total = collections.Counter()
    print(f"{'ternary weights':<16}{'changed':>12}{'0<->+-1':>12}{'sign flip':>12}")
    for group in sorted(flips):
        c = flips[group]
        total.update(c)
        n = c["n"]
        print(f"{group:<16}{c['changed'] / n:>12.3e}{(c['zero_to_nonzero'] + c['nonzero_to_zero']) / n:>12.3e}{c['sign_flips'] / n:>12.3e}")
    n = total["n"]
    print(f"{'ALL':<16}{total['changed'] / n:>12.3e}   ({total['changed']:,} of {n:,} codes)")

    print("\nfull-precision tensors, relative L2 change:")
    by_group = collections.defaultdict(list)
    for name, rel in other.items():
        by_group[group_of(name)].append(rel)
    for group in sorted(by_group):
        vals = by_group[group]
        print(f"  {group:<28} mean {sum(vals) / len(vals):.3e}  max {max(vals):.3e}  ({len(vals)} tensors)")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(per_tensor, f, indent=1)


if __name__ == "__main__":
    main()

"""Pre-training sanity check on one GPU: vLLM BitNet rollout vs. HF transformers (the FSDP actor's model).

1. Samples responses with vLLM (T=1, like training) and records the sampled tokens' log-probs.
2. Re-scores the same tokens with transformers' BitNetForCausalLM (AutoBitLinear, online QAT), which
   is exactly what the verl actor computes, and reports the log-prob mismatch.
3. Greedy GSM8K accuracy with and without the leading <|begin_of_text|> (the custom chat template).

Usage (from the BitNet repo root, inside the verl container):
    PYTHONPATH=$PWD python -m agenticrl.scripts.check_rollout_parity \
        --model models/bitnet-b1.58-2B-4T-bf16 --data data/gsm8k_verl/test.parquet
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG = os.path.join(REPO_ROOT, "agenticrl", "config", "grpo_gsm8k_bitnet.yaml")


def build_prompts(tokenizer, df: pd.DataFrame, template: str) -> list[list[int]]:
    tokenizer.chat_template = template
    prompts = []
    for messages in df["prompt"]:
        ids = tokenizer.apply_chat_template([dict(m) for m in messages], add_generation_prompt=True, tokenize=True)
        prompts.append(list(ids["input_ids"] if not isinstance(ids, list) else ids))
    return prompts


@torch.no_grad()
def hf_token_logprobs(model, prompt: list[int], response: list[int]) -> np.ndarray:
    ids = torch.tensor([prompt + response], device=model.device)
    logits = model(ids).logits[0, len(prompt) - 1 : -1].float()
    target = ids[0, len(prompt) :]
    return logits.log_softmax(-1).gather(-1, target[:, None])[:, 0].cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=os.path.join(REPO_ROOT, "models", "bitnet-b1.58-2B-4T-bf16"))
    parser.add_argument("--data", default=os.path.join(REPO_ROOT, "data", "gsm8k_verl", "test.parquet"))
    parser.add_argument("--num-prompts", type=int, default=64, help="prompts for the log-prob parity check")
    parser.add_argument("--num-eval", type=int, default=500, help="prompts for the greedy BOS / no-BOS accuracy check")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.4)
    parser.add_argument("--enforce-eager", action="store_true", help="disable vLLM torch.compile / CUDA graphs")
    parser.add_argument("--out", default=None, help="optional JSON file for the summary")
    args = parser.parse_args()

    # vLLM's engine-core / worker subprocesses must be able to import agenticrl.
    os.environ["PYTHONPATH"] = REPO_ROOT + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
    import agenticrl.bitnet  # noqa: F401  (registers BitNet with vLLM, patches transformers)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    from agenticrl.rewards.gsm8k import compute_score

    with open(CONFIG) as f:
        template = yaml.safe_load(f)["actor_rollout_ref"]["model"]["custom_chat_template"]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    stock_template = tokenizer.chat_template
    df = pd.read_parquet(args.data)
    eval_df = df.iloc[: args.num_eval]
    prompts_bos = build_prompts(tokenizer, eval_df, template)
    prompts_nobos = build_prompts(tokenizer, eval_df, stock_template)
    assert prompts_bos[0][0] == tokenizer.bos_token_id and prompts_nobos[0][0] != tokenizer.bos_token_id

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=2048,
        enforce_eager=args.enforce_eager,
        seed=0,
    )
    summary = {}

    # ---- 1+2: sampled rollouts, vLLM vs HF log-probs ----
    sample_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=args.max_tokens, logprobs=0, seed=0)
    parity_prompts = prompts_bos[: args.num_prompts]
    t0 = time.time()
    outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in parity_prompts], sample_params)
    print(f"[vllm] sampled {len(outs)} responses in {time.time() - t0:.1f}s")

    hf = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    q_proj = hf.model.layers[0].self_attn.q_proj
    print(f"[hf] q_proj={type(q_proj).__name__} online_quant={getattr(q_proj, 'online_quant', None)} lm_head={type(hf.lm_head).__name__}")

    diffs, vllm_lps, hf_lps = [], [], []
    for prompt, out in zip(parity_prompts, outs, strict=True):
        completion = out.outputs[0]
        response = list(completion.token_ids)
        if not response:
            continue
        v_lp = np.array([lp[tok].logprob for lp, tok in zip(completion.logprobs, response, strict=True)])
        h_lp = hf_token_logprobs(hf, prompt, response)
        diffs.append(v_lp - h_lp)
        vllm_lps.append(v_lp)
        hf_lps.append(h_lp)
    d = np.concatenate(diffs)
    v_all, h_all = np.concatenate(vllm_lps), np.concatenate(hf_lps)
    prob_diff = np.abs(np.exp(v_all) - np.exp(h_all))
    seq_mean_abs = np.array([np.abs(x).mean() for x in diffs])
    summary["parity"] = {
        "tokens": int(d.size),
        "logprob_abs_diff_mean": float(np.abs(d).mean()),
        "logprob_abs_diff_p99": float(np.percentile(np.abs(d), 99)),
        "logprob_abs_diff_max": float(np.abs(d).max()),
        "logprob_diff_mean (vllm-hf)": float(d.mean()),
        "prob_abs_diff_mean": float(prob_diff.mean()),
        "prob_pearson": float(np.corrcoef(np.exp(v_all), np.exp(h_all))[0, 1]),
        "worst_seq_mean_abs_diff": float(seq_mean_abs.max()),
    }
    del hf
    torch.cuda.empty_cache()

    # ---- 3: greedy accuracy, with vs. without BOS ----
    greedy = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    for name, prompts in (("with_bos", prompts_bos), ("without_bos", prompts_nobos)):
        outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], greedy)
        scores = [
            compute_score("openai/gsm8k", o.outputs[0].text, gt)
            for o, gt in zip(outs, (r["ground_truth"] for r in eval_df["reward_model"]), strict=True)
        ]
        summary[f"greedy_{name}"] = {k: float(np.mean([s[k] for s in scores])) for k in ("acc", "acc_flexible", "has_marker")}
        summary[f"greedy_{name}"]["mean_response_tokens"] = float(np.mean([len(o.outputs[0].token_ids) for o in outs]))
        if name == "with_bos":
            print("\n[example greedy response]\n" + outs[0].outputs[0].text[:1200] + "\n")

    print(json.dumps(summary, indent=2))
    p = summary["parity"]
    # verl's own rollout/actor mismatch for well-supported bf16 models is typically ~1e-2 mean |dlogp|.
    verdict = "OK" if p["logprob_abs_diff_mean"] < 0.03 and abs(p["logprob_diff_mean (vllm-hf)"]) < 0.01 else "CHECK"
    print(f"[parity] {verdict}: mean |dlogp| = {p['logprob_abs_diff_mean']:.4f}, bias = {p['logprob_diff_mean (vllm-hf)']:+.4f}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()

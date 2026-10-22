"""Summarize verl GRPO runs from their TensorBoard logs into one short report to share.

    python -m agenticrl.tools.analyze_run outputs/tensorboard/agenticrl/h20x2_lr1e-5 \
        --parity parity.json --log train.log

``tb_dir`` is one run (a dir with ``events.out.tfevents.*``) or a parent of several runs, e.g.
``outputs/tensorboard/agenticrl`` for a learning-rate sweep, which adds a comparison table. Writes
``run_summary.md`` (send this) and ``scalars_<run>.csv`` (every scalar, every step) to ``--out-dir``.
Step 0 is the validation before training; training steps start at 1.
"""

import argparse
import json
import math
import os
import re
import statistics

from tensorboard.backend.event_processing import event_accumulator

VAL_ACC = re.compile(r"^val-core/.+/acc/mean@\d+$")
# (label, regex over TensorBoard tags); the first matching tag is reported.
VAL_METRICS = [
    ("acc", r"^val-core/.+/acc/mean@\d+$"),
    ("acc_flexible", r"^val-aux/.+/acc_flexible/mean@\d+$"),
    ("has_marker", r"^val-aux/.+/has_marker/mean@\d+$"),
]
TRAIN_METRICS = [
    ("reward (train batch)", r"^critic/score/mean$"),
    ("kl_loss", r"^actor/kl_loss$"),
    ("ppo_kl", r"^actor/ppo_kl$"),
    ("pg_clipfrac", r"^actor/pg_clipfrac$"),
    ("entropy", r"^actor/entropy$"),
    ("grad_norm", r"^actor/grad_norm$"),
    ("lr", r"^actor/lr$"),
    ("response_len mean", r"^response_length/mean$"),
    ("response_len clip_ratio", r"^response_length/clip_ratio$"),
    ("rollout_probs_diff mean", r"^training/rollout_probs_diff_mean$"),
    ("rollout_probs_diff max", r"^training/rollout_probs_diff_max$"),
    ("rollout/actor probs pearson", r"^training/rollout_actor_probs_pearson_corr$"),
    ("step time (s)", r"^timing_s/step$"),
    ("throughput (tok/s/gpu)", r"^perf/throughput$"),
    ("max GPU mem alloc (GB)", r"^perf/max_memory_allocated_gb$"),
]


def find_runs(tb_dir: str) -> dict[str, str]:
    runs = {}
    for root, _, files in os.walk(tb_dir):
        if any(f.startswith("events.out.tfevents") for f in files):
            runs[os.path.relpath(root, tb_dir) if root != tb_dir else os.path.basename(os.path.normpath(tb_dir))] = root
    if not runs:
        raise SystemExit(f"no TensorBoard event files under {tb_dir}")
    return dict(sorted(runs.items()))


def load_run(path: str):
    acc = event_accumulator.EventAccumulator(
        path, size_guidance={event_accumulator.SCALARS: 0, event_accumulator.TENSORS: 0}
    )
    acc.Reload()
    scalars = {}
    for tag in acc.Tags()["scalars"]:
        series = {}
        for event in acc.Scalars(tag):  # a resumed run re-logs some steps: keep the last write
            series[event.step] = event.value
        scalars[tag] = dict(sorted(series.items()))
    texts = {}
    # torch's SummaryWriter.add_text stores "<tag>" as the tensor "<tag>/text_summary".
    text_tag = next((t for t in acc.Tags().get("tensors", []) if t.startswith("val/generations")), None)
    if text_tag is not None:
        for event in acc.Tensors(text_tag):
            texts[event.step] = event.tensor_proto.string_val[0].decode("utf-8", "replace")
    return scalars, texts


def pick(scalars: dict, pattern: str):
    for tag in sorted(scalars):
        if re.match(pattern, tag):
            return tag
    return None


def finite(values):
    return [v for v in values if not (math.isnan(v) or math.isinf(v))]


def fmt(v, pct=False):
    if v is None:
        return "-"
    if math.isnan(v) or math.isinf(v):
        return str(v)
    if pct:
        return f"{100 * v:.2f}%"
    return f"{v:.4g}"


def train_row(label, series):
    steps = [s for s in series if s >= 1]
    vals = [series[s] for s in steps]
    good = finite(vals)
    if not good:
        return f"| {label} | - | - | - | - | {len(vals) - len(good)} non-finite |"
    k = max(1, min(3, len(good) // 4 or 1))
    first, last = statistics.fmean(good[:k]), statistics.fmean(good[-k:])
    lo_step = steps[vals.index(min(good))]
    hi_step = steps[vals.index(max(good))]
    bad = len(vals) - len(good)
    note = f"{bad} non-finite" if bad else ""
    return (
        f"| {label} | {fmt(first)} | {fmt(last)} | {fmt(min(good))} @{lo_step} | {fmt(max(good))} @{hi_step} | {note} |"
    )


def health_checks(scalars, val_size) -> list[str]:
    """Heuristic flags; each says what to look at, none of them is a verdict on its own."""
    flags = []

    def series(pattern):
        tag = pick(scalars, pattern)
        if tag is None:
            return None
        s = {k: v for k, v in scalars[tag].items() if k >= 1}
        return finite(list(s.values())) or None

    def ends(vals):
        k = max(1, min(3, len(vals) // 4 or 1))
        return statistics.fmean(vals[:k]), statistics.fmean(vals[-k:])

    acc_tag = pick(scalars, VAL_ACC.pattern)
    if acc_tag and 0 in scalars[acc_tag] and len(scalars[acc_tag]) > 1:
        for label, pattern in VAL_METRICS[:2]:
            tag = pick(scalars, pattern)
            if tag is None or 0 not in scalars[tag]:
                continue
            p0, p1 = scalars[tag][0], list(scalars[tag].values())[-1]
            se = math.sqrt(max(p0 * (1 - p0) + p1 * (1 - p1), 1e-12) / val_size)
            z = (p1 - p0) / se
            verdict = "超出噪声范围" if abs(z) >= 2 else "在噪声范围内（|z|<2），不能确认有提升"
            flags.append(
                f"val {label}: {fmt(p0, True)} -> {fmt(p1, True)}（{100 * (p1 - p0):+.2f} pt，约 {z:+.1f}σ，{verdict}）"
            )
        marker_tag = pick(scalars, VAL_METRICS[2][1])
        if marker_tag and 0 in scalars[marker_tag]:
            m0, m1 = scalars[marker_tag][0], list(scalars[marker_tag].values())[-1]
            if abs(m1 - m0) > 0.2:
                flags.append(
                    f"⚠ has_marker {fmt(m0, True)} -> {fmt(m1, True)}：答题格式变化很大，acc（严格判分）的提升"
                    "大部分来自格式；推理能力的变化看 acc_flexible"
                )
    elif acc_tag and 0 not in scalars[acc_tag]:
        flags.append("没有 step 0 的验证结果（val_before_train 关了？），无法和训练前比较")

    kl = series(r"^actor/kl_loss$")
    if kl:
        if max(kl) < 1e-4:
            flags.append(f"⚠ kl_loss 始终 < 1e-4（max {max(kl):.2e}）：策略几乎没离开初始模型，学习率可能偏小")
        else:
            flags.append(f"kl_loss 最终 {kl[-1]:.3g}（max {max(kl):.3g}）：策略确实在变化")

    diff = series(r"^training/rollout_probs_diff_mean$")
    if diff:
        a, b = ends(diff)
        msg = f"rollout/actor 概率差 mean: {a:.4g} -> {b:.4g}"
        if b > 2 * a and b > 0.01:
            msg = "⚠ " + msg + "：持续变大，检查 vLLM 插件 / 权重同步"
        flags.append(msg)
    corr = series(r"^training/rollout_actor_probs_pearson_corr$")
    if corr and min(corr) < 0.99:
        flags.append(f"⚠ rollout/actor 概率相关系数最低 {min(corr):.4f}（<0.99）")

    ent = series(r"^actor/entropy$")
    if ent:
        a, b = ends(ent)
        if b < 0.3 * a:
            flags.append(f"⚠ entropy {a:.3g} -> {b:.3g}：下降超过 70%，可能熵崩塌（探索不足）")
        elif b > 2 * a:
            flags.append(f"⚠ entropy {a:.3g} -> {b:.3g}：翻倍以上，输出可能在变乱")

    gn = series(r"^actor/grad_norm$")
    if gn and len(gn) >= 5:
        med = statistics.median(gn)
        if max(gn) > 10 * med:
            flags.append(f"⚠ grad_norm 突刺：max {max(gn):.3g} > 10× 中位数 {med:.3g}")
    for label, pattern in TRAIN_METRICS:
        tag = pick(scalars, pattern)
        if tag and len(finite(scalars[tag].values())) < len(scalars[tag]):
            flags.append(f"⚠ {label} 出现 NaN/Inf")

    clip = series(r"^response_length/clip_ratio$")
    if clip and clip[-1] > 0.05:
        flags.append(f"⚠ 回答被截断比例 {clip[-1]:.1%}（>5%）：考虑加大 max_response_length")

    score = series(r"^critic/score/mean$")
    if score and score[-1] > 0.9:
        flags.append(f"训练 batch 奖励已到 {score[-1]:.2f}：大部分题组 8 个回答都对，GRPO 优势为 0，学习信号在减少")
    return flags


def extract_samples(text: str, limit: int, chars: int) -> str:
    parts = re.split(r"(?m)^### Sample \d+\n", text)[1 : limit + 1]
    out = []
    for i, part in enumerate(parts, 1):
        part = part.replace("---", "").strip()
        part = re.sub(r"\n{3,}", "\n\n", part)
        out.append(f"**Sample {i}**\n\n" + (part if len(part) <= chars else part[:chars] + " …[截断]"))
    return "\n\n".join(out)


def write_csv(path, scalars):
    steps = sorted({s for series in scalars.values() for s in series})
    tags = sorted(scalars)
    with open(path, "w") as f:
        f.write("step," + ",".join(tags) + "\n")
        for s in steps:
            f.write(str(s) + "," + ",".join(repr(scalars[t][s]) if s in scalars[t] else "" for t in tags) + "\n")


def summarize_run(name, scalars, texts, args) -> list[str]:
    steps = sorted({s for series in scalars.values() for s in series})
    lines = [f"## {name}", "", f"- tags: {len(scalars)}，steps: {steps[0]}–{steps[-1]}（训练步 {sum(s >= 1 for s in steps)}）", ""]

    lines += ["### 验证集", ""]
    val_tags = [(label, pick(scalars, p)) for label, p in VAL_METRICS]
    val_tags = [(label, t) for label, t in val_tags if t]
    if val_tags:
        val_steps = sorted({s for _, t in val_tags for s in scalars[t]})
        lines.append("| step | " + " | ".join(label for label, _ in val_tags) + " |")
        lines.append("|---" * (len(val_tags) + 1) + "|")
        for s in val_steps:
            lines.append(f"| {s} | " + " | ".join(fmt(scalars[t].get(s), pct=True) for _, t in val_tags) + " |")
    else:
        lines.append("（没有 val-core / val-aux 指标）")
    lines.append("")

    lines += ["### 训练指标", "", "| 指标 | 开头 | 结尾 | 最小 | 最大 | 备注 |", "|---|---|---|---|---|---|"]
    for label, pattern in TRAIN_METRICS:
        tag = pick(scalars, pattern)
        if tag:
            lines.append(train_row(label, scalars[tag]))
    lines.append("")

    n_train = sum(s >= 1 for s in steps)
    timing = {}
    for tag, series in scalars.items():
        if tag.startswith("timing_s/"):
            vals = finite([v for s, v in series.items() if s >= 1])
            if vals:
                timing[tag[len("timing_s/") :]] = (statistics.fmean(vals), statistics.median(vals), len(vals))
    if "step" in timing:
        _, step_median, _ = timing["step"]
        every = sorted(((k, v) for k, v in timing.items() if k != "step" and v[2] >= n_train), key=lambda kv: -kv[1][0])
        sometimes = sorted(((k, v) for k, v in timing.items() if k != "step" and v[2] < n_train), key=lambda kv: -kv[1][0])
        lines += [
            "### 每步耗时（秒）",
            "",
            f"step 中位数 {step_median:.1f}s（均值 {timing['step'][0]:.1f}s）；每步都有的阶段（均值）："
            + "，".join(f"{k} {v[0]:.1f}" for k, v in every[:6]),
        ]
        if sometimes:
            lines.append("只在部分步出现（每次均值，出现次数）：" + "，".join(f"{k} {v[0]:.1f}（{v[2]} 步）" for k, v in sometimes[:6]))
        lines.append("")

    lines += ["### 检查", ""] + [f"- {flag}" for flag in health_checks(scalars, args.val_size)] + [""]

    if texts and args.samples > 0:
        first, last = min(texts), max(texts)
        lines += ["### 验证样例", ""]
        for s in dict.fromkeys([first, last]):
            lines += [f"#### step {s}", "", extract_samples(texts[s], args.samples, args.sample_chars), ""]
    return lines


def comparison(runs_data, val_size) -> list[str]:
    lines = [
        "## 实验对比",
        "",
        "| run | step0 acc | 最终 acc | Δ (pt) | 最佳 acc @step | acc_flexible step0 → 最终 (Δ pt) | 最终 kl_loss | 训练步 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, (scalars, _) in runs_data.items():
        tag = pick(scalars, VAL_ACC.pattern)
        flex_tag = pick(scalars, VAL_METRICS[1][1])
        kl_tag = pick(scalars, r"^actor/kl_loss$")
        if not tag:
            lines.append(f"| {name} | - | - | - | - | - | - | - |")
            continue
        series = scalars[tag]
        p0, p1 = series.get(0), list(series.values())[-1]
        best_step = max(series, key=series.get)
        delta = f"{100 * (p1 - p0):+.2f}" if p0 is not None else "-"
        flex = "-"
        if flex_tag:
            f0, f1 = scalars[flex_tag].get(0), list(scalars[flex_tag].values())[-1]
            flex = f"{fmt(f0, True)} → {fmt(f1, True)}" + (f" ({100 * (f1 - f0):+.2f})" if f0 is not None else "")
        kl = fmt(list(scalars[kl_tag].values())[-1]) if kl_tag else "-"
        n_steps = sum(s >= 1 for s in {s for v in scalars.values() for s in v})
        lines.append(
            f"| {name} | {fmt(p0, True)} | {fmt(p1, True)} | {delta} | {fmt(series[best_step], True)} @{best_step} "
            f"| {flex} | {kl} | {n_steps} |"
        )
    se = math.sqrt(2 * 0.75 * 0.25 / val_size)
    lines += ["", f"参考：acc≈75%、{val_size} 题时，两次评测差值的标准误约 {100 * se:.1f} pt。", ""]
    return lines


def parity_section(path) -> list[str]:
    with open(path) as f:
        parity = json.load(f)
    lines = ["## parity.json（训练前自检）", ""]
    for section, values in parity.items():
        lines.append(f"- **{section}**: " + "，".join(f"{k}={fmt(v) if isinstance(v, float) else v}" for k, v in values.items()))
    return lines + [""]


def log_section(path) -> list[str]:
    patterns = {
        "Initial validation metrics": r"Initial validation metrics",
        "Final validation metrics": r"Final validation metrics",
        "Traceback": r"Traceback \(most recent call last\)",
        "CUDA OOM": r"CUDA out of memory|OutOfMemoryError",
        "NaN/Inf": r"\bnan\b|\binf\b",
        "WARNING": r"\bWARNING\b|UserWarning",
    }
    counts = dict.fromkeys(patterns, 0)
    first_lines = {}
    with open(path, errors="replace") as f:
        for line in f:
            for key, pattern in patterns.items():
                if re.search(pattern, line, flags=re.IGNORECASE if key == "NaN/Inf" else 0):
                    counts[key] += 1
                    first_lines.setdefault(key, line.strip()[:300])
    lines = ["## 控制台日志", ""] + [f"- {k}: {v} 行" for k, v in counts.items()]
    for key in ("Initial validation metrics", "Traceback", "CUDA OOM"):
        if key in first_lines:
            lines.append(f"- 第一处 {key}: `{first_lines[key]}`")
    return lines + [""]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tb_dir", help="one run's TensorBoard dir, or a parent of several runs")
    parser.add_argument("--parity", default=None, help="parity.json from scripts/check_rollout_parity.py")
    parser.add_argument("--log", default=None, help="console log of the training run (optional)")
    parser.add_argument("--out-dir", default="run_analysis")
    parser.add_argument("--val-size", type=int, default=1319, help="validation set size (GSM8K test: 1319)")
    parser.add_argument("--samples", type=int, default=2, help="validation samples to show at the first/last step")
    parser.add_argument("--sample-chars", type=int, default=1500)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    runs = find_runs(args.tb_dir)
    runs_data = {name: load_run(path) for name, path in runs.items()}

    report = ["# 训练日志分析", ""]
    if len(runs_data) > 1:
        report += comparison(runs_data, args.val_size)
    for name, (scalars, texts) in runs_data.items():
        if not scalars:
            report += [f"## {name}", "", "（没有标量数据）", ""]
            continue
        report += summarize_run(name, scalars, texts, args)
        write_csv(os.path.join(args.out_dir, f"scalars_{re.sub(r'[^A-Za-z0-9_.-]+', '_', name)}.csv"), scalars)
    if args.parity:
        report += parity_section(args.parity)
    if args.log:
        report += log_section(args.log)

    text = "\n".join(report)
    out = os.path.join(args.out_dir, "run_summary.md")
    with open(out, "w") as f:
        f.write(text)
    print(text)
    print(f"\n[analyze_run] wrote {out} and {len(runs_data)} CSV file(s) to {args.out_dir}/")


if __name__ == "__main__":
    main()

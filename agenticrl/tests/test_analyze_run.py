import json
import subprocess
import sys

import pytest

pytest.importorskip("tensorboard")
from torch.utils.tensorboard import SummaryWriter  # noqa: E402

ACC = "val-core/openai/gsm8k/acc/mean@1"
FLEX = "val-aux/openai/gsm8k/acc_flexible/mean@1"
MARKER = "val-aux/openai/gsm8k/has_marker/mean@1"


def write_run(path, acc0, acc_end, kl_scale, nan_step=None, steps=29, flex=None, marker=(0.98, 0.98)):
    """Mimic verl's TensorBoard output: val at step 0 and every 5 steps, train metrics from step 1."""
    w = SummaryWriter(str(path))
    for s in [0, 5, 10, 15, 20, 25, steps]:
        t = s / steps
        acc = acc0 + (acc_end - acc0) * t
        w.add_scalar(ACC, acc, s)
        w.add_scalar(FLEX, acc + 0.001 if flex is None else flex[0] + (flex[1] - flex[0]) * t, s)
        w.add_scalar(MARKER, marker[0] + (marker[1] - marker[0]) * t, s)
        w.add_text("val/generations", f"**Generation Results - Step {s}**\n\n### Sample 1\n**Input:** q{s}\n\n**Output:** #### {s}\n\n**Score:** 1.0\n\n---\n\n", s)
    for s in range(1, steps + 1):
        w.add_scalar("critic/score/mean", 0.5 + 0.01 * s, s)
        w.add_scalar("actor/kl_loss", kl_scale * s, s)
        w.add_scalar("actor/entropy", 0.6, s)
        w.add_scalar("actor/grad_norm", float("nan") if s == nan_step else 0.3, s)
        w.add_scalar("training/rollout_probs_diff_mean", 0.004, s)
        w.add_scalar("timing_s/step", 300.0, s)
        w.add_scalar("timing_s/gen", 90.0, s)
    w.add_scalar("timing_s/save_checkpoint", 97.0, steps)  # verl logs it only on save steps
    w.close()


def test_report(tmp_path):
    write_run(tmp_path / "tb" / "lr1e-5", acc0=0.60, acc_end=0.75, kl_scale=1e-3, nan_step=7)
    write_run(tmp_path / "tb" / "lr1e-6", acc0=0.60, acc_end=0.605, kl_scale=1e-7)
    parity = tmp_path / "parity.json"
    parity.write_text(json.dumps({"parity": {"logprob_abs_diff_mean": 0.012}, "greedy_with_bos": {"acc": 0.6}}))
    out = tmp_path / "out"
    proc = subprocess.run(
        [sys.executable, "-m", "agenticrl.tools.analyze_run", str(tmp_path / "tb"), "--parity", str(parity), "--out-dir", str(out)],
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    report = (out / "run_summary.md").read_text()

    assert "## 实验对比" in report
    assert "| lr1e-5 | 60.00% | 75.00% | +15.00 |" in report
    assert "| lr1e-6 | 60.00% | 60.50% | +0.50 |" in report
    assert "60.10% → 75.10% (+15.00)" in report  # acc_flexible column
    assert "超出噪声范围" in report and "不能确认有提升" in report  # +15 pt vs +0.5 pt
    assert "kl_loss 始终 < 1e-4" in report  # only the lr=1e-6 run
    assert report.count("kl_loss 始终 < 1e-4") == 1
    assert "grad_norm 出现 NaN/Inf" in report
    assert "#### step 0" in report and "#### step 29" in report and "q29" in report
    assert "logprob_abs_diff_mean=0.012" in report
    assert "save_checkpoint 97.0（1 步）" in report and "gen 90.0" in report
    assert "has_marker" not in report.split("### 检查")[1].split("###")[0]  # format unchanged: no warning

    csv = (out / "scalars_lr1e-5.csv").read_text().splitlines()
    assert csv[0].startswith("step,") and len(csv) == 1 + 30  # header + steps 0..29


def test_format_dominated_gain(tmp_path):
    # The real h20x2_lr1e-5 run: strict acc 5% -> 75%, but the base model already solved 68% without "####".
    write_run(tmp_path / "tb" / "run", acc0=0.0523, acc_end=0.7506, kl_scale=1e-3, flex=(0.6801, 0.7513), marker=(0.072, 0.984))
    out = tmp_path / "out"
    proc = subprocess.run(
        [sys.executable, "-m", "agenticrl.tools.analyze_run", str(tmp_path / "tb" / "run"), "--out-dir", str(out)],
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    checks = (out / "run_summary.md").read_text().split("### 检查")[1].split("###")[0]
    assert "val acc: 5.23% -> 75.06%" in checks
    assert "val acc_flexible: 68.01% -> 75.13%（+7.12 pt" in checks
    assert "⚠ has_marker 7.20% -> 98.40%" in checks and "推理能力的变化看 acc_flexible" in checks

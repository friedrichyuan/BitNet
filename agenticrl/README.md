# agenticrl：基于 verl 的 BitNet b1.58 强化学习

第一个实验：**GRPO + GSM8K**，模型为 `models/bitnet-b1.58-2B-4T-bf16`（bf16 latent 权重，训练用版本）。
Actor/Ref 用 verl 的 FSDP（transformers 原生 `BitNetForCausalLM`，`AutoBitLinear` 在线 QAT），Rollout 用 vLLM 0.24。

```
agenticrl/
├── bitnet/                  # BitNet 适配层（import 即生效）
│   ├── quant.py             #   三值权重 / int8 激活的 fake-quant 数学（actor 与 rollout 共用）
│   ├── hf_patch.py          #   transformers 侧修正：QAT 算子改 eager；去掉无效 auto_map
│   ├── vllm_quant.py        #   vLLM 量化方法 "bitnet"：每个 Linear 的输入做 int8 per-token 量化
│   └── vllm_model.py        #   vLLM 的 BitNetForCausalLM（加载权重时逐张量三值化）
├── data/gsm8k.py            # datasets/gsm8k/main/*.parquet -> verl RL 数据格式
├── rewards/gsm8k.py         # 规则奖励（#### 后的数字）+ 诊断指标 acc_flexible / has_marker
├── config/grpo_gsm8k_bitnet.yaml   # 叠加在 verl ppo_trainer.yaml 之上的训练配置
├── scripts/
│   ├── run_grpo_gsm8k_bitnet.sh    # 训练启动脚本（所有关键超参可用环境变量覆盖）
│   └── check_rollout_parity.py     # 训练前 GPU 自检：vLLM vs HF 的 log-prob 一致性 + BOS 对比
├── tools/ternary_drift.py   # 训练后分析：三值权重翻转比例、全精度参数相对变化
└── tests/                   # CPU 单元测试（容器内可跑，无需 GPU）
```

## 在 HPC 上运行

以下命令都在 BitNet 仓库根目录、verl 容器内执行。`VERL_ROOT` 指向 verl 源码（容器里默认 `/workspace/verl`；若 verl 已 pip 安装可忽略）。

### 0. CPU 单元测试（可选，约 2 分钟）

```bash
export PYTHONPATH=$PWD:${VERL_ROOT:-/workspace/verl}
python -m pytest agenticrl/tests -q
```

### 1. 训练前自检（1 张 GPU，约 5–10 分钟）—— 请先跑这个

```bash
PYTHONPATH=$PWD python -m agenticrl.data.gsm8k --input_dir datasets/gsm8k/main --output_dir datasets/gsm8k_verl
PYTHONPATH=$PWD python -m agenticrl.scripts.check_rollout_parity --out parity.json
```

它会输出：
- `parity`：vLLM 采样的 token，用 HF 模型（即 actor）重新打分后的 log-prob 差异。预期 `logprob_abs_diff_mean` 在 1e-2 量级、偏差（`vllm-hf`）接近 0，最后一行打印 `OK` 或 `CHECK`。
- `greedy_with_bos` / `greedy_without_bos`：500 道测试题的贪心准确率，用来确认加 BOS 的模板是对的。

### 2. 训练

```bash
# 默认：单机 8 卡，lr=1e-5，256 prompts/step × 8 samples，5 epochs（约 145 steps）
bash agenticrl/scripts/run_grpo_gsm8k_bitnet.sh

# 常用覆盖
NGPUS_PER_NODE=4 ACTOR_LR=3e-5 EXPERIMENT_NAME=lr3e-5 bash agenticrl/scripts/run_grpo_gsm8k_bitnet.sh
# 只打印最终配置、不训练
bash agenticrl/scripts/run_grpo_gsm8k_bitnet.sh --cfg job
# 也可以在末尾追加任意 hydra 覆盖，例如 trainer.total_epochs=3
```

输出默认在 `outputs/`：checkpoint 在 `outputs/checkpoints/`（每个约 35 GB，最多保留 2 个），
TensorBoard 日志在 `outputs/tensorboard/<project>/<experiment>/`（可用 `TENSORBOARD_DIR` 改）。指标只写 TensorBoard 和控制台，不用 wandb。
多机：先 `ray start` 组好集群，再设 `NNODES`。

查看曲线（HPC 上开 TensorBoard，本地用 SSH 端口转发访问 http://localhost:6006）：

```bash
tensorboard --logdir outputs/tensorboard --port 6006 --bind_all
ssh -L 6006:<计算节点>:6006 <登录节点>      # 在本地机器上执行
```

不同实验（如学习率扫描）用不同的 `EXPERIMENT_NAME`，会落在同一 `--logdir` 下的不同子目录，TensorBoard 里可以直接叠加对比。
验证集样例回答（`log_val_generations: 10`）在 TensorBoard 的 TEXT 页。

**建议的学习率扫描**（见下文"学习率"）：`ACTOR_LR=1e-6`（对照组，预期几乎不变）、`1e-5`、`3e-5`。

### 3. 训练后分析权重变化

```bash
PYTHONPATH=$PWD python -m agenticrl.tools.ternary_drift \
    --base models/bitnet-b1.58-2B-4T-bf16 \
    --target outputs/checkpoints/agenticrl/<exp>/global_step_<N>/actor/huggingface
```

`actor/huggingface` 是普通的 HF 目录（配置里开了 `hf_model` 导出），也能直接被 vLLM / transformers 加载。

## 需要关注的指标

| 指标 | 含义 |
|---|---|
| `val-core/openai/gsm8k/acc/mean@1` | 测试集贪心准确率（主指标；step 0 为训练前基线） |
| `val-aux/openai/gsm8k/acc_flexible/mean@1` | 只看最后一个数字是否正确；与 `acc` 的差距 = 格式问题 |
| `val-aux/openai/gsm8k/has_marker/mean@1` | 回答里出现 `#### 数字` 的比例 |
| `critic/score/mean` | 训练 batch 平均奖励 |
| `actor/kl_loss` | 与初始模型的 KL：**一直接近 0 说明模型基本没变**（学习率太小的典型症状） |
| `training/rollout_probs_diff_mean` / `_max` | vLLM 与 FSDP actor 的概率差（训练中的一致性监控） |
| `actor/entropy`, `response_length/mean`, `response_length/clip_ratio` | 熵崩塌 / 回答变长 / 被截断 |

## 关键设计与原因

1. **vLLM 不支持 BitNet**（0.24 里没有这个架构，也不认识 `quant_method: bitnet`）。`agenticrl/bitnet` 补了一个 vLLM 模型和量化方法，数学与 transformers 的 `AutoBitLinear(online)` 完全一致：
   - 权重在**加载时**按原始张量（q/k/v、gate/up 各自）做 absmean 三值化，之后才被 vLLM 合并成 qkv/gate_up；verl 每步同步权重也走同一路径（fp32 master 权重会先转 bf16 再三值化，与 actor 前向一致）。
   - 每个 Linear 的输入做 int8 per-token absmax 量化；attn/ffn sub-norm、relu² 与 HF 相同。
   - CPU 测试中 fp32 下与 HF **逐 token 完全一致**（KL ≈ 1e-9）；bf16 下只有舍入顺序带来的噪声，真实模型的数值由第 1 步自检给出。
   - 仅支持 rollout `tensor_model_parallel_size=1`（sub-norm 和激活 absmax 都需要完整的 hidden 维；2B 模型单卡足够）。
2. **注册方式**：vLLM 的 server actor 和 worker 子进程不会执行 `external_lib`，但都会 `import verl`；配置里通过 Ray `runtime_env` 设置 `VERL_USE_EXTERNAL_MODULES=agenticrl.bitnet` 和 `PYTHONPATH`，让每个进程都完成注册。**无需修改 verl 或 vLLM 源码。**
3. **BOS**：BitNet 自带的 chat template 不输出 `<|begin_of_text|>`，而 verl 用 `add_special_tokens=False` 编码模板，所以默认 prompt 没有 BOS。配置里的 `custom_chat_template` = 原模板前加 `{{ bos_token }}`。
4. **Checkpoint 崩溃修复**：模型 `config.json` 里的 `auto_map` 指向仓库里不存在的 remote code；verl 保存 checkpoint 时据此调用 `custom_object_save`，会在**第一次保存时报 FileNotFoundError**。`hf_patch` 在构建 `BitNetConfig` 时丢弃这个 `auto_map`（模型本身走 transformers 原生实现，不受影响）。
5. **QAT 算子改 eager**：transformers 上游用 `@torch.compile` 包了量化算子，在 verl 的动态 batch + 梯度检查点下会频繁重编译；eager 版本与上游逐位一致（有测试）。设 `AGENTICRL_BITNET_HF_COMPILE=1` 可恢复上游行为。
6. **镜像与 verl 版本差异**：`verlai/verl:vllm024.dev2` 对应 2026 年 7–9 月的 verl（vLLM 0.24 / torch 2.11 / TransferQueue 0.1.8），而当前 verl main 是按 vLLM 0.29 / torch 2.13 / TransferQueue 0.1.11 测试的（官方用 `uv run` 安装匹配依赖）。已知差异：TQ 0.1.8 不接受 `total_storage_size: null`，配置里固定为 100000（verl 当时的默认值）；TQ 0.1.8 没有 `save_checkpoint`，只影响异步训练模式，我们用的是同步模式。若再遇到版本相关报错，可选：`pip install --no-deps TransferQueue==0.1.11`，或把 verl 切到与镜像同期的提交 `2d98ec77`。

## 学习率（最重要的超参）

BitNet 的 latent 权重幅度很大（各层 mean|w| ≈ 1.2–2.8），而 FSDP 前向时先把 fp32 master 权重转成 bf16 再三值化。在三值阈值（0.5·mean|w|）附近，bf16 的步长约 0.004–0.008。
verl 常用的 `lr=1e-6` 下，Adam 每步每个权重只移动约 1e-6，145 步累计约 1.5e-4，**几乎不会有三值权重翻转**，模型只能通过 embedding/RMSNorm 微小变化——训练可能"看起来在跑但什么都没学到"。
因此默认 `lr=1e-5`，并建议做上面的扫描；`actor/kl_loss` 与 `ternary_drift` 的翻转比例可以直接验证权重是否真的在变。

## 请反馈给我

1. `parity.json`（或第 1 步的完整输出）。
2. 训练指标：TensorBoard 目录（`outputs/tensorboard/...`，体积很小，可直接打包发我）或截图，至少包括上表各项，以及 step 0 的验证结果。
3. 报错时：完整 traceback 和启动命令（含环境变量覆盖）。
4. 若有 checkpoint：`ternary_drift` 的输出。
5. 机器信息：GPU 型号/数量、每步耗时（`timing_s/step`）。

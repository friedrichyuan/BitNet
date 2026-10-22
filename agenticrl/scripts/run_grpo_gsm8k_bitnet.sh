#!/usr/bin/env bash
# GRPO | BitNet b1.58 2B-4T (bf16 latent weights) | GSM8K | FSDP actor + vLLM rollout (TP=1)
#
# Every knob below can be overridden from the environment, and extra hydra overrides can be
# appended, e.g.
#   ACTOR_LR=3e-5 EXPERIMENT_NAME=lr3e-5 bash agenticrl/scripts/run_grpo_gsm8k_bitnet.sh trainer.total_epochs=3
# Print the composed config without training:
#   bash agenticrl/scripts/run_grpo_gsm8k_bitnet.sh --cfg job

set -xeuo pipefail

# ---- paths ----
BITNET_ROOT=${BITNET_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
VERL_ROOT=${VERL_ROOT:-/workspace/verl}       # verl source checkout; ignored if verl is pip-installed
MODEL_PATH=${MODEL_PATH:-${BITNET_ROOT}/models/bitnet-b1.58-2B-4T-bf16}
RAW_DATA_DIR=${RAW_DATA_DIR:-${BITNET_ROOT}/datasets/gsm8k/main}
DATA_DIR=${DATA_DIR:-${BITNET_ROOT}/datasets/gsm8k_verl}
PROJECT_NAME=${PROJECT_NAME:-agenticrl}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-grpo_gsm8k_bitnet2b}
OUTPUT_DIR=${OUTPUT_DIR:-${BITNET_ROOT}/outputs}  # checkpoints + tensorboard; checkpoints are ~35 GB each
CKPT_DIR=${CKPT_DIR:-${OUTPUT_DIR}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}

# ---- cluster ----
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

# ---- algorithm / batch ----
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-256}       # prompts per step
ROLLOUT_N=${ROLLOUT_N:-8}                       # responses per prompt (GRPO group size)
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}  # prompts per optimizer step
ACTOR_LR=${ACTOR_LR:-1e-5}
# Separate lr for the q/k/v/o/gate/up/down latent weights (ACTOR_LR then only applies to RMSNorm
# weights + embeddings). Empty = one lr for everything. Switches the actor to FSDP2. See README.
BITLINEAR_LR=${BITLINEAR_LR:-}
USE_KL_LOSS=${USE_KL_LOSS:-True}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}
TEMPERATURE=${TEMPERATURE:-1.0}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-5}
TEST_FREQ=${TEST_FREQ:-5}
SAVE_FREQ=${SAVE_FREQ:-50}

# ---- memory ----
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.5}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-12288}
LOG_PROB_MAX_TOKEN_LEN_PER_GPU=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-16384}

LOGGER=${LOGGER:-'["console","tensorboard"]'}  # metrics go to TensorBoard only; wandb is never imported

# agenticrl and verl must be importable in the driver, every Ray worker and every vLLM subprocess.
export PYTHONPATH="${BITNET_ROOT}:${VERL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export VERL_USE_EXTERNAL_MODULES=agenticrl.bitnet
export TENSORBOARD_DIR=${TENSORBOARD_DIR:-${OUTPUT_DIR}/tensorboard/${PROJECT_NAME}/${EXPERIMENT_NAME}}
echo "TensorBoard logs: ${TENSORBOARD_DIR}  (view: tensorboard --logdir ${OUTPUT_DIR}/tensorboard)"

BITLINEAR_ARGS=()
if [ -n "${BITLINEAR_LR}" ]; then
    BITLINEAR_ARGS=(
        actor_rollout_ref.actor.strategy=fsdp2
        "actor_rollout_ref.actor.optim.override_optimizer_config={bitlinear_lr:${BITLINEAR_LR}}"
    )
fi

if [ ! -f "${DATA_DIR}/train.parquet" ] || [ ! -f "${DATA_DIR}/test.parquet" ]; then
    python3 -m agenticrl.data.gsm8k --input_dir "${RAW_DATA_DIR}" --output_dir "${DATA_DIR}"
fi

python3 -m verl.trainer.main_ppo \
    --config-path="${BITNET_ROOT}/agenticrl/config" \
    --config-name=grpo_gsm8k_bitnet \
    data.train_files="${DATA_DIR}/train.parquet" \
    data.val_files="${DATA_DIR}/test.parquet" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.use_kl_loss="${USE_KL_LOSS}" \
    actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}" \
    actor_rollout_ref.actor.entropy_coeff="${ENTROPY_COEFF}" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}" \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.temperature="${TEMPERATURE}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL}" \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU}" \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU}" \
    reward.custom_reward_function.path="${BITNET_ROOT}/agenticrl/rewards/gsm8k.py" \
    reward.custom_reward_function.name=compute_score \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.logger="${LOGGER}" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.TENSORBOARD_DIR="${TENSORBOARD_DIR}" \
    trainer.nnodes="${NNODES}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.save_freq="${SAVE_FREQ}" \
    "${BITLINEAR_ARGS[@]}" \
    "$@"

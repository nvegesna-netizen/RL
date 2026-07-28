#!/bin/bash

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
readonly SBATCH_PARTITION="batch_block1,backfill"

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "Required environment variable is unset: ${name}" >&2
    exit 2
  fi
}

for name in \
  CONTAINER \
  MOUNTS \
  NEMO_RL_CHAT_TEMPLATE \
  NEMO_RL_MODEL \
  NEMO_RL_VIDEO_TRAIN_JSONL \
  NEMO_RL_VIDEO_VAL_JSONL \
  NEMO_RL_VIDEO_MEDIA_ROOT \
  NEMO_RL_RUN_ROOT \
  SBATCH_ACCOUNT \
  WANDB_API_KEY \
  WANDB_ENTITY \
  WANDB_PROJECT; do
  require_env "${name}"
done

case "${SBATCH_ACCOUNT}" in
  nemotron_edge_omni | nemotron_omni_vision) ;;
  *)
    echo "SBATCH_ACCOUNT must be nemotron_edge_omni or nemotron_omni_vision" >&2
    exit 2
    ;;
esac

mkdir -p "${NEMO_RL_RUN_ROOT}"

export BASE_LOG_DIR="${NEMO_RL_RUN_ROOT}"
export CONTAINER
export GPUS_PER_NODE=8
export MOUNTS
export NEMO_RL_CHAT_TEMPLATE
export NEMO_RL_MODEL
export NEMO_RL_REPO="${REPO_ROOT}"
export NEMO_RL_RUN_ROOT
export NEMO_RL_VIDEO_MEDIA_ROOT
export NEMO_RL_VIDEO_TRAIN_JSONL
export NEMO_RL_VIDEO_VAL_JSONL
export NRL_RUN_PREFIX="rl_main_vg16_async_video_tp4_unlimited"
export WANDB_API_KEY
export WANDB_ENTITY
export WANDB_PROJECT

read -r -d '' COMMAND <<'COMMAND_EOF' || true
set -euo pipefail

source /opt/venv/bin/activate
export PATH="/opt/venv/bin:${PATH}"
export HF_HOME="${NEMO_RL_RUN_ROOT}/hf_home"
export NRL_FORCE_REBUILD_VENVS=true
export NRL_VIDEO_BACKEND=torchcodec
export NRL_VIDEO_SAMPLING_STYLE=nemotron_vl
export NRL_VIDEO_SFT_MAX_FRAMES=32
export NRL_VIDEO_SFT_MIN_FRAMES=32
export NRL_VIDEO_TEMPORAL_PATCH_SIZE=2
export TORCH_CUDA_ARCH_LIST=9.0
export VLLM_MAMBA_BACKEND=flashinfer
export VLLM_RAY_EXTRA_ENV_VARS_TO_COPY="PYTHONPATH,NEMO_RL_VIDEO_MEDIA_ROOT,NRL_VIDEO_BACKEND,NRL_VIDEO_SAMPLING_STYLE,NRL_VIDEO_TEMPORAL_PATCH_SIZE"
export VLLM_VIDEO_LOADER_BACKEND=nemotron_vl
unset NRL_IGNORE_VERSION_MISMATCH

readonly RUN_NAME="${NRL_RUN_PREFIX}_${SLURM_JOB_ID}"
readonly CHECKPOINT_DIR="${NEMO_RL_RUN_ROOT}/checkpoints/${RUN_NAME}"
mkdir -p "${CHECKPOINT_DIR}" "${HF_HOME}"
cd "${NEMO_RL_REPO}"

uv run --project /opt/nemo-rl --no-sync python \
  examples/nemo_gym/run_grpo_nemo_gym.py \
  --config examples/nemo_gym/grpo_nemotron_omni_30ba3b_video_async.yaml \
  policy.model_name="${NEMO_RL_MODEL}" \
  policy.tokenizer.name="${NEMO_RL_MODEL}" \
  policy.tokenizer.chat_template="${NEMO_RL_CHAT_TEMPLATE}" \
  policy.generation.vllm_cfg.http_server_serving_chat_kwargs.chat_template="${NEMO_RL_CHAT_TEMPLATE}" \
  grpo.max_num_steps=-1 \
  grpo.max_num_epochs=1000000 \
  grpo.seq_logprob_error_threshold=null \
  grpo.num_prompts_per_step=16 \
  grpo.num_generations_per_prompt=16 \
  grpo.async_grpo.enabled=true \
  grpo.async_grpo.max_trajectory_age_steps=1 \
  grpo.async_grpo.in_flight_weight_updates=true \
  policy.train_global_batch_size=256 \
  policy.max_total_sequence_length=32768 \
  policy.megatron_cfg.tensor_model_parallel_size=4 \
  policy.megatron_cfg.expert_model_parallel_size=8 \
  policy.generation.max_new_tokens=16000 \
  policy.generation.vllm_cfg.tensor_parallel_size=4 \
  policy.generation.vllm_cfg.max_model_len=32768 \
  policy.generation.vllm_kwargs.max_num_batched_tokens=32768 \
  policy.generation.vllm_kwargs.max_num_seqs=1 \
  policy.generation.colocated.enabled=false \
  policy.generation.colocated.resources.num_nodes=12 \
  cluster.num_nodes=16 \
  cluster.gpus_per_node=8 \
  checkpointing.enabled=true \
  checkpointing.checkpoint_dir="${CHECKPOINT_DIR}" \
  logger.log_dir="${NEMO_RL_RUN_ROOT}/${RUN_NAME}/training" \
  logger.wandb_enabled=true \
  logger.wandb.project="${WANDB_PROJECT}" \
  +logger.wandb.entity="${WANDB_ENTITY}" \
  logger.wandb.name="${RUN_NAME}"
COMMAND_EOF
export COMMAND

cd "${REPO_ROOT}"
sbatch \
  --account="${SBATCH_ACCOUNT}" \
  --partition="${SBATCH_PARTITION}" \
  --nodes=16 \
  --gpus-per-node=8 \
  --time="${SBATCH_TIME:-04:00:00}" \
  --job-name=rl_main_vg16_async_video \
  --output="${NEMO_RL_RUN_ROOT}/slurm-16n-async-%j.out" \
  ray.sub

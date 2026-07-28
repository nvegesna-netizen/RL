# Nemotron Omni video GRPO launchers

This directory contains the reproducible Slurm launchers used to validate
video Gym training:

- `run_2n_sync.sh`: 2-node synchronous GRPO on the `interactive` partition
  and `nemotron_edge_omni` account.
- `run_16n_async.sh`: 16-node asynchronous GRPO on either the
  `nemotron_edge_omni` or `nemotron_omni_vision` account.

The canonical training and data configuration is kept with the NeMo RL
recipes:

- `examples/nemo_gym/grpo_nemotron_omni_30ba3b_video_sync.yaml`
- `examples/nemo_gym/grpo_nemotron_omni_30ba3b_video_async.yaml`
- `examples/nemo_gym/prepare_video_dataset.py`

Both launchers use cached-video JSONL input. They intentionally keep
`grpo.max_num_steps=-1` and `grpo.seq_logprob_error_threshold=null`; the
four-hour Slurm allocation, rather than a GRPO step cap, ends a validation
run. They also enable W&B and checkpointing. No credential, user-specific
path, temporary source overlay, or version-mismatch bypass is embedded in
the scripts.

## Prepare the cached-video dataset

The recipe expects training and validation JSONL files plus a media root.
The JSONL may be generated from a supported source dataset with:

```bash
uv run examples/nemo_gym/prepare_video_dataset.py --help
```

Set `NEMO_RL_VIDEO_MEDIA_ROOT` to the filesystem prefix from which the video
paths in the JSONL can be resolved. No dataset content is committed to this
repository.

## Required environment

Set these variables before launching either job:

```bash
export CONTAINER=/path/to/nemo-rl-vllm-0.25.1.sqsh
export MOUNTS=/lustre:/lustre
export NEMO_RL_MODEL=/path/to/nemotron-omni-checkpoint
export NEMO_RL_CHAT_TEMPLATE="${NEMO_RL_MODEL}/chat_template.jinja"
export NEMO_RL_VIDEO_TRAIN_JSONL=/path/to/cached_video_train.jsonl
export NEMO_RL_VIDEO_VAL_JSONL=/path/to/cached_video_validation.jsonl
export NEMO_RL_VIDEO_MEDIA_ROOT=/lustre
export NEMO_RL_RUN_ROOT=/path/to/training-output
export WANDB_API_KEY=...
export WANDB_ENTITY=...
export WANDB_PROJECT=...
```

The container must provide the NeMo RL environment at `/opt/nemo-rl`, use
the supported vLLM version, and contain the video decoding dependencies.
Credentials must come from the caller's environment or an approved secret
mechanism; do not add them to these scripts.

## Launch 2-node synchronous validation

```bash
bash ehsan_scripts/run_2n_sync.sh
```

The launcher submits a four-hour job. To change only the Slurm duration,
set `SBATCH_TIME` before invoking it. Do not use the duration to introduce a
GRPO step limit.

## Launch 16-node asynchronous validation

Select one approved account and launch:

```bash
export SBATCH_ACCOUNT=nemotron_edge_omni
bash ehsan_scripts/run_16n_async.sh
```

`SBATCH_ACCOUNT=nemotron_omni_vision` is also supported. If jobs are raced
between accounts, cancel the duplicate immediately after one starts.

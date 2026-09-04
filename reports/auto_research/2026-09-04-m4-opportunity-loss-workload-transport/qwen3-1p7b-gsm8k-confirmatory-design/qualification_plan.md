# Qwen3-1.7B GSM8K neutral qualification plan

Status: `FROZEN_LOCAL_DESIGN_PENDING_PACKAGE_AND_EXPLICIT_RELEASE`.

## Purpose

Run exactly 32 trainer steps on the frozen two-GPU non-colocated topology to
measure GSM8K resource fit, information yield, throughput, reward support, and
observer duty before considering the fixed 558-step confirmatory acquisition.
Controlled release is disabled; its inert configuration contains only a
`neutral` zero-delay arm for observer metadata. This qualification is
operational training, but it is not randomized causal acquisition and its
observations are forbidden from the confirmatory estimator.

The qualification inherits the preflighted Qwen3-1.7B/GSM8K overlay. It changes
only the run length, qualification observer identity, disabled neutral release
configuration, and output paths. Model, tokenizer, GRPO settings, prompt,
dataset, processor, verifier, sampler, audit semantics, and two-GPU topology
remain fixed.

## Frozen runtime contract

- config:
  `examples/configs/grpo_math_1B_megatron_single_controller_m4_qwen3_1p7b_gsm8k_neutral_qualification.yaml`;
- trainer steps: 32 exactly;
- GPUs: two on one EOS node, one generator and one trainer;
- controlled release: disabled, with one inert `neutral` arm at 0.0 seconds;
- observer assignment domain:
  `m4-opportunity-loss-qwen3-1p7b-gsm8k-neutral-qualification-v1`;
- dataset: `openai/gsm8k`, subset `main`, split `train`;
- automatic retry and extension: false;
- scheduler ceiling: four wall-clock hours and eight GPU-hours, although the
  qualification is expected to finish much earlier.

The terminal-green no-training evidence is compute artifact SHA-256
`94c21c7d8fea68e7c698df94d52c153ca6c7ce324edf0c87fa951c30d63d389d`,
lock SHA-256
`5ced4c617c5f0154a2b03b89262583e94b884b6de2f683470cbac984eff99991`,
and summary SHA-256
`354ed332c97842054d09ea39fab00c863a44bd9b3a80ee461f5f085c7cbbf8d0`.

## Fail-closed qualification gates

Qualification is green only if all of the following hold:

1. The package, preflight artifact, lock, summary, source, config, protocol, and
   image hashes match their frozen values.
2. The run completes exactly 32 ordered trainer versions on two GPUs with no
   checkpoint resume, retry, extension, or second submission.
3. Controlled release remains disabled and no release-assignment, delay-start,
   or delay-completion lifecycle event occurs.
4. Opportunity coverage is complete, rewards are binary, and both reward values
   occur. Group IDs are unique and no control-versus-delay estimate is computed.
5. Corrected observer duty is at most 0.01 and the observation count equals the
   number of opportunity groups.
6. The conservative opportunity-group rate is at least 15 groups per observed start
   version, projecting at least 7,500 assignments over the already-fixed 500
   primary versions and therefore exceeding the registered minimum of 7,395.
7. Projected 558-step wall time, computed as measured fixed overhead plus 558
   times the qualification's mean completed-step duration and multiplied by a
   frozen 1.25 safety factor, is no more than four hours. The corresponding
   two-GPU projection must be no more than eight GPU-hours.

Any failed or missing gate yields `QUALIFICATION_RED` and stops the sequence.
The qualification cannot resize the 558-step window, relax a threshold, trigger
an automatic retry, or authorize confirmatory acquisition.

## Release boundary

This local design does not authorize EOS submission or qualification training.
After a deterministic package and review receipt are frozen, one separate
explicit user release is required. Any launcher must use `runllm.py --no_wait`.

# Qwen3-1.7B GSM8K neutral qualification plan

Status: `R1_TERMINAL_INCOMPLETE_FINITE_EPOCH_EXHAUSTION`.

## Purpose

Run exactly 32 trainer steps on the frozen two-GPU non-colocated topology to
measure GSM8K resource fit, information yield, throughput, reward support, and
observer duty before considering the fixed 558-step confirmatory acquisition.
The controlled-release assignment instrument is enabled with exactly one
`neutral` zero-delay arm. It emits the assignment metadata required by the
opportunity audit while applying no intervention and creating no treatment
contrast. This qualification is operational training, but it is not randomized
causal acquisition and its observations are forbidden from the confirmatory
estimator.

The qualification inherits the preflighted Qwen3-1.7B/GSM8K overlay. It changes
only the run length, qualification observer identity, zero-dose neutral release
configuration, and output paths. Model, tokenizer, GRPO settings, prompt,
dataset, processor, verifier, sampler, audit semantics, and two-GPU topology
remain fixed.

## Frozen runtime contract

- config:
  `examples/configs/grpo_math_1B_megatron_single_controller_m4_qwen3_1p7b_gsm8k_neutral_qualification.yaml`;
- trainer steps: 32 exactly;
- GPUs: two on one EOS node, one generator and one trainer;
- controlled-release instrument: enabled, with one `neutral` arm at exactly
  0.0 seconds and therefore no release-delay intervention;
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
3. The assignment instrument remains enabled with exactly one `neutral` arm at
   0.0 seconds. Every assignment record has neutral label, zero delay, arm mass
   one, and total mass one. The completed opportunity-group IDs match the
   delay-start and delay-completion group IDs; extra terminal assignments may
   exist only as a superset of completed group IDs.
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

## Operational amendment

R2 showed that disabling the assignment instrument while enabling the
opportunity audit is rejected by the authoritative single-controller config
validator. The accepted Qwen3-1.7B neutral qualification used the supported
one-arm, zero-dose pattern and produced the required metadata without a causal
contrast. This amendment changes only the operational instrument switch and
the corresponding lifecycle gate. It does not change the frozen control:d5
confirmatory protocol, estimator, thresholds, geometry, or acquisition status.
The historical `protocol_config.json` qualification metadata remains immutable
at its registered hash; its `controlled_release_enabled: false` field is
superseded for qualification execution only by this amendment.

## Bootstrap amendment after R5

R5 passed the provenance and config gates and initialized both GPU workers, but
failed before its first trainer step because its isolated run tree lacked the
container's submodule contents. A Git archive carries the submodule mount-point
directories, not the checked-out Megatron files. The accepted Qwen3-1.7B
qualification avoided this by copying the pinned `/opt/nemo-rl` tree before
overlaying frozen source.

The R6 packaging repair adopts that accepted ordering and adds a fail-closed
pre-training check that the dependency fingerprint matches the container and
that `megatron` imports from the isolated run tree. The source archive, config,
instrument, model, data, run length, thresholds, and scientific protocol do not
change. R5 artifact SHA-256 is
`43aff533273fa96b0a3df523307fe4b6a1730ebcd197819d91363598c9a0ef47`;
the authorization-independent R6 package contract SHA-256 is
`0d448ade4b31a22f595c9df23b2cd43d78aac3c5ddca75c34546f07e23a37a6d`.
The distinct R6 release was subsequently authorized and submitted exactly once
through `runllm.py --no_wait` as parent pipeline `66330380`; manifest SHA-256 is
`514c369a0fa68dd110cff29806760092362df21467d5e46ecf3d3d78515ffc83`.
The one-shot guard is consumed. This submission does not authorize any retry,
extension, or confirmatory acquisition.

## Scheduler amendment after R6

R6 never received an allocation. Slurm `5978609` remained pending for priority
for `04:00:19`, accumulated zero runtime, and was canceled as `DEADLINE` when
JET's fixed eight-hour deadline could no longer fit the requested four-hour
runtime. No workload gate or training code executed.

A successor qualification should request 90 minutes from Slurm while retaining
the registered four-hour confirmatory ceiling. This is an operational scheduler
repair, not a scientific change: it gives priority queueing up to approximately
6.5 hours within JET's deadline and remains above both the accepted
qualification's 984.588-second workload duration and R5's observed
initialization-to-first-result window. R6 authority is consumed, and this
amendment authorizes no R7 submission.

The distinct R7 release was subsequently authorized and submitted exactly once
through `runllm.py --no_wait` as parent pipeline `66390680`. Frozen manifest
SHA-256 is
`0b532760600e290c65b524b5ba258a3e007c7568c5148fcf60ed4c89d4cfd1a8`.
The one-shot guard is consumed; no retry, extension, or confirmatory acquisition
is authorized.

## R7 terminal result

R7 is terminal green. Parent `66390680`, downstream `66390878`, EOS compute
`427180439`, and Slurm `5980958` succeeded without retry. The fingerprint and
Megatron bootstrap repair passed, all 32 steps completed, and all eight frozen
qualification gates passed. The run produced 534 complete opportunity groups
across 32 start versions, corrected observer duty 0.00260494, and binary reward
support. It projects the fixed 558-step design at 1.68598 wall-hours and 3.37196
GPU-hours.

Qualification data remain excluded from the causal estimator. This result
permits separately freezing the confirmatory acquisition package but does not
authorize its submission.

## Confirmatory package freeze

The authorization-independent package contract is now frozen at SHA-256
`a36972e575f31606d2fc38477fe4461ef76e42dc0d410efac4f5e90a041bae40`.
Local validation reconciled the source archive, replacement-preflight artifact
and 31/2 delta attestation, R7 result and terminal artifact, protocol,
control:d5 acquisition config, 558-step geometry, resource caps, and the
GSM8K-specific analysis entrypoint. The builder fails closed without a new
exact acquisition authorization. No manifest, one-shot guard, submission, or
acquisition exists at this boundary.

The package requests 2.5 hours from Slurm, compared with R7's conservative
1.68598-hour projection, and retains the registered four-hour/eight-GPU-hour
scientific cap. Automatic retry and extension remain false. A future authorized
submission must use `runllm.py --no_wait` and must not reuse any historical
qualification or OpenMath acquisition authority.

The distinct acquisition was subsequently authorized and submitted exactly
once through `runllm.py --no_wait`. Parent pipeline `66461092`, generator job
`427764393`, downstream pipeline `66461158`, and exact EOS compute job
`427764750` are recorded in `acquisition_submission.json`. The one-shot guard is
consumed. The attempt subsequently failed at the pre-acquisition Megatron import
gate before any training or scientific ledger was created; retry or extension
remains forbidden. See `acquisition_attempt_1_failure.md` for the reconciled
terminal evidence and scientific interpretation.

The authorized R1 repair restores R7's required `import nemo_rl` then
`import megatron` bootstrap order. Its rendered scientific execution tail is
byte-identical to attempt 1. It was submitted exactly once through
`runllm.py --no_wait`: parent `66508153`, successful generator `428138575`,
downstream `66508187`, and exact `jet-eos` compute job `428138883`. The R1
one-shot guard is consumed. That checkpoint established launch identity only;
the terminal result is reconciled below. No scientific input changed, and retry
or extension remains forbidden; see `acquisition_r1_bootstrap_repair.json`.

R1 subsequently proved that the qualification's local throughput and
information-yield extrapolation was incomplete. The one-epoch GSM8K stream
ended after 394 complete steps, leaving three groups where four were required.
Its strict primary join contains 7,290 assignments, below the frozen 7,395
minimum, and it never entered the terminal-guard window. The qualification did
not check finite-dataset capacity after stale eviction; its earlier green status
therefore remains historical evidence for bootstrap, two-GPU fit, reward
support, observer duty, and short-run throughput only—not for 558-step
one-epoch completion. See `acquisition_r1_terminal_failure.json`.

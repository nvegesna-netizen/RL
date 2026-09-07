# Qwen3-1.7B GSM8K M4 confirmatory workload-transport design

Status: `R1_TERMINAL_INCOMPLETE_FINITE_EPOCH_EXHAUSTION`.
The replacement preflight and terminal-green R7 qualification are recorded
separately. The resource and information-yield gates support the now-frozen
confirmatory package contract. This document does not authorize EOS submission
or scientific acquisition.

## Scientific question

Does a fixed five-second release delay materially increase pre-release
opportunity loss for Qwen3-1.7B on GSM8K under the accepted M4 instrument and
estimator?

The intended scientific change from the completed Qwen3-1.7B study is the
dataset: explicit `openai/gsm8k`, subset `main`, split `train`. The model, GRPO
contract, prompt, verifier, topology, release arms, replay semantics, estimator,
thresholds, and fixed acquisition geometry remain unchanged.

The inherited OpenMath validation split is disabled (`split_validation_size=0`)
and its inactive split seed is explicitly null, because GSM8K directly supplies
the registered training split and does not consume those OpenMath split controls.

## Frozen geometry

- burn-in start versions: 0–7;
- primary start versions: 8–507;
- terminal guard start versions: 508–557;
- fixed trainer steps: 558;
- minimum primary assignments: 7,395;
- preferred primary assignments: 7,500;
- topology: one generation GPU plus one trainer GPU;
- hard scheduler cap: four wall-clock hours and eight GPU-hours.

The window is fixed before GSM8K qualification. Qualification may gate the run
on feasibility and information yield, but it cannot resize the window from
treatment outcomes or enter the confirmatory estimator.

## Analysis and stopping

The primary estimator remains the accepted eight-fold cross-fitted
generalized-regression adjustment for opportunity and its zero indicator.
Inference remains clustered by start weight version with four-lag Bartlett HAC,
eight-version circular blocks, 20,000 bootstrap draws, and the outer interval
envelope. Materiality requires the lower envelope endpoint to exceed
`Delta_L=0.2` at alpha 0.05. The original unadjusted estimator is supporting
only. The mechanism check is a support condition, and corrected observer duty is
an external-validity qualifier with a 1% ceiling.

There is no outcome-guided stopping, automatic extension, rerandomization, or
automatic retry. Falling short of 7,395 scored primary assignments is reported
and blocks a decisive claim; it does not authorize additional steps.

## Gate sequence

1. A dedicated static protocol validator must bind the dataset-only overlay,
   protocol, audit, accepted instrument, and output identities.
2. A pinned containerized no-training preflight must resolve the exact config,
   run the complete selected test suite, and prove that no acquisition began.
3. A separately authorized neutral 32-step qualification may test GSM8K
   throughput, reward support, completeness, and observer duty with the
   supported enabled one-arm zero-dose instrument and causal analysis forbidden.
4. The fixed 558-step acquisition may be considered only if qualification
   projects at least 7,395 primary assignments and completion within four hours.

Every gate is fail-closed. A failed gate stops the sequence; it does not weaken
the protocol.

## Frozen confirmatory package boundary

The authorization-independent acquisition package contract has SHA-256
`a36972e575f31606d2fc38477fe4461ef76e42dc0d410efac4f5e90a041bae40`.
It binds source commit `be0a56c37542f81a6413287fc110f291c07c5156`, source
archive SHA-256
`0130ed6a8af2ad9a37b8dad5d4b44c4cb72e5a5f43a1a6ac5d1066f99e652901`,
protocol SHA-256
`84c43e4fc32a7f647505729a556d5e0fb088a38fd7eedd1ea235dd51a4ec3fc7`,
and terminal R7 artifact SHA-256
`9e66f58c4a005fdb33673eb6741bb9b254bd5bc46f17828b7f5ec9ca559e51c4`.

The package invokes the GSM8K-specific analyzer
`tools/opportunity_loss_workload_transport_pipeline.py` (SHA-256
`9a99af42e759ffb15ba095fecfcc0e07870ee47aa54e09e871636471e45344e3`),
not the historical OpenMath transport analyzer. The authorization-gated builder
has SHA-256
`492377992a2a64b66a628e1ab620f76a015170dd93eb45745f233d8e9d21ed63`.
It requires the accepted
container-baseline/source-overlay bootstrap, exact fingerprint equality, a
Megatron import from the isolated run tree, at least 7,395 primary assignments,
and explicit exclusion of qualification observations from the estimator.

The requested Slurm duration is 2.5 hours. This is above the R7 projection of
1.68598 hours while leaving queueing room inside JET's fixed eight-hour
deadline; the registered four-wall-hour/eight-GPU-hour scientific ceiling is
unchanged. Retrying and extending remain disabled. No acquisition manifest,
submission guard, or EOS launch was created while freezing this contract. A
fresh, exact acquisition authorization is the next required gate, and any
authorized launcher must use `runllm.py --no_wait`.

## Confirmatory acquisition submission

The user subsequently authorized exactly one acquisition bound to the frozen
contract. Authorization SHA-256 is
`2bd233d897d4bedcb1be8416476005a1dc3770a4f5efa6a1abf615eab7362ccf`;
the validated, authorization-bound manifest SHA-256 is
`18b817ec76f99b158010f1031f47f0e02c75912fe6b0896afde9e1649f80b9c5`.
It was submitted once through `runllm.py --no_wait` as parent pipeline
`66461092`. Generator job `427764393` succeeded and created downstream pipeline
`66461158`. The exact two-GPU EOS compute identity is job `427764750`,
`basic/m4-qwen3-1p7b-gsm8k-confirmatory-control-d5-acquisition dgxh100_eos 00
[2 dgxh100_eos]`.

The one-shot guard is consumed. The attempt terminated before acquisition, so
no causal or materiality conclusion is available. No retry or extension is
authorized.

## Attempt 1 terminal result

Parent pipeline `66461092`, downstream pipeline `66461158`, and EOS compute job
`427764750` terminated failed; generator `427764393`, logs-before `427764749`,
and logs-after `427764751` succeeded. The compute artifact is 47,241,819 bytes
with SHA-256
`971c8479e55e5d4031ce6d09fe8c9dbff41f817d71c4ee76f6509c31c2716f01`.
Slurm job `5983488` received an allocation after approximately 1 hour 23
minutes of queueing. The workload script then failed within about five seconds
of starting its package gates, with rank exit code 1.

All frozen input hashes and safe archive extraction passed. The next bootstrap
check raised `ModuleNotFoundError: No module named 'megatron'`. The acquisition
builder imported `megatron` directly, whereas the terminal-green R7 builder
first imported `nemo_rl`, which registers the vendored Megatron path, and then
imported `megatron`. This is a package-gate import-order omission, not a model,
workload, intervention, estimator, or resource failure.

The fingerprint/Megatron pass marker, frozen-authority pass, config pass,
acquisition-start marker, and 558-step completion marker are all absent. The
artifact also contains no acquisition run log, lifecycle ledger, opportunity
ledger, observer-duty record, causal result, summary, or checksum ledger. Thus
zero confirmatory observations entered an estimator, and this attempt provides
no evidence for or against materiality on GSM8K. The R7 qualification remains
green; the GSM8K confirmatory question remains open. Any operational repair
would require a new package, validation that exactly reproduces the accepted
R7 import order, and fresh explicit authorization. It would not be an automatic
retry or a scientific protocol change.

## R1 bootstrap repair

The user authorized a narrow successor repair and one EOS rerun. The repaired
builder adds exactly the missing `import nemo_rl` before `import megatron`,
matching terminal-green R7 and the path-registration behavior implemented in
`nemo_rl/__init__.py`. No scientific runtime file, source archive, protocol,
config, arm, assignment, window, estimator, threshold, or resource cap changed.

The repaired package contract SHA-256 is
`d36354e77bd5f206cc11f41d5a08b9182572096a0f451f9cf95b1e02ec9b09c0`;
authorization SHA-256 is
`aced07031b2969222948db6ca0e916f1c6c149f8be5ea03ae88d56b1aa508e16`;
and the validated manifest SHA-256 is
`65100dfa1b3a0b8525e32b9498d321f0dc2a66db52a202bd9d03a76227295c17`.
Validation parsed Bash and all five embedded Python blocks, proved the standalone
bootstrap lines are exactly `import nemo_rl` then `import megatron`, confirmed a
single training entrypoint, and proved the entire scientific execution tail from
the acquisition-start marker onward is byte-identical to attempt 1. The distinct
R1 guard was absent at the pre-submission checkpoint. The package was then
submitted exactly once through `runllm.py --no_wait`: parent pipeline
`66508153`, successful generator job `428138575`, downstream pipeline
`66508187`, and EOS compute job `428138883`. The compute job name exactly
matches the R1 workload name and carries the `jet-eos` tag. The guard is now
consumed. At that checkpoint this verified launch identity, not scientific
success; the terminal result is reconciled below.

## R1 terminal result

Parent `66508153`, downstream `66508187`, and EOS compute job `428138883`
terminated failed; generator `428138575`, logs-before `428138882`, and
logs-after `428138884` succeeded. Slurm job `5985036` ran for `01:06:50` and
failed with exit `1:0`, well before its 2.5-hour limit. The logs-after artifact
is 125,905,140 bytes with SHA-256
`dae0b7ee0f13df156c205a336f2c5ff14cfff7093a92e337d8a88ecb25b14785`.

The bootstrap repair worked and all frozen package gates passed. Training
completed 394/558 steps before the inherited `max_num_epochs: 1` exhausted the
finite 7,472-group opportunity stream. The controller then correctly rejected
the three residual groups because a complete step requires four:
`dispatched 0/4 prompt groups with 3 group(s) remaining in the buffer`.
Accounting closes exactly at 1,576 selected groups, 5,893 stale evictions, and
three cancelled residual groups.

The preserved ledgers contain 112,474 lifecycle rows, 7,472 opportunity-group
rows, and 394 train-step completion rows. Strict frozen joining yields only
7,290 primary assignments over start versions 8–393 (`control=3,673`,
`d5=3,617`), 105 below the preregistered minimum of 7,395; no terminal-guard
version 508–557 exists. Corrected observer duty is 0.00301711. Because the
completion and minimum-information gates both failed, no canonical result,
summary, or checksum ledger was emitted. R1 is `INCOMPLETE`: it supports
neither a material nor a non-material GSM8K conclusion, and its outcomes must
not enter a successor confirmatory estimator.

The operational defect is now the prospective capacity assumption, not the
bootstrap, scheduler, model, or GPU runtime. The 32-step qualification
extrapolated per-version yield without checking that one GSM8K epoch could
supply the fixed 558-step window after stale eviction. Repeating the unchanged
one-epoch package would deterministically fail again.

## Prospective R2 capacity repair boundary

The narrow capacity-safe successor is a new protocol, not a retry: retain the
558 steps, windows, minimum assignment count, arms, estimator, thresholds,
model, workload, and resource cap, but explicitly permit two GSM8K epochs and
use a fresh assignment domain and seed. R1 observed 394 complete steps from one
epoch; two epochs therefore provide a 230-step empirical margin over the
558-step target at the observed selection rate. The bounded-run controller
cancels the rollout pump when step 558 completes, so it need not consume all of
epoch two.

The R2 protocol must explicitly define repeated prompt exposures as distinct
randomized group instances, exclude every R1 observation, bind a capacity gate
to the two-epoch setting, and independently validate the unchanged inference
contract. No R2 package, authority, guard, or launch exists at this checkpoint.
See `acquisition_r1_terminal_failure.json` and
`acquisition_r2_capacity_repair_plan.json`.

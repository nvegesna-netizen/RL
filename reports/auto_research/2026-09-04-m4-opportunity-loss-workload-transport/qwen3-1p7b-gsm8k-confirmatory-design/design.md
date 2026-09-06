# Qwen3-1.7B GSM8K M4 confirmatory workload-transport design

Status: `CONFIRMATORY_ACQUISITION_SUBMITTED_AWAITING_TERMINAL_RESULT`.
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

The one-shot guard is consumed. The acquisition is not yet terminal, so no
causal or materiality conclusion is available. No retry or extension is
authorized.

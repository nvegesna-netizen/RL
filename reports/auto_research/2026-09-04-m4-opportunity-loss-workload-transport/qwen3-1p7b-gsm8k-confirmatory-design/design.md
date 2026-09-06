# Qwen3-1.7B GSM8K M4 confirmatory workload-transport design

Status: `QUALIFICATION_GREEN_PENDING_CONFIRMATORY_PACKAGE_AND_AUTHORIZATION`.
The replacement preflight and terminal-green R7 qualification are recorded
separately. The resource and information-yield gates support freezing the fixed
confirmatory package. This document does not authorize EOS submission or
scientific acquisition.

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

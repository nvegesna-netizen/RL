# Qwen3-1.7B M4 confirmatory transport design

Status: `FROZEN_LOCAL_PROTOCOL_PENDING_NO_TRAINING_PREFLIGHT`. This document
does not authorize training or scientific acquisition.

## Scientific question

Does a fixed five-second release delay materially increase pre-release
opportunity loss for Qwen3-1.7B on OpenMathInstruct-2 under the accepted M4
instrument and estimator?

This changes only model scale relative to the completed Qwen3-0.6B study. The
workload, GRPO loss contract, control:d5 allocation, delay, lifecycle semantics,
opportunity replay, primary estimator, missingness bounds, material threshold,
mechanism support condition, and observer-duty ceiling remain unchanged.

## Prospective geometry

The neutral qualification completed 32/32 steps on the smallest two-GPU
non-colocated topology. It produced 544 opportunity groups, 17.0 per observed
version and 16.4333 per interior version. Planning deliberately floors this to
15 assignments per primary version.

- burn-in start versions: 0–7;
- primary start versions: 8–507 (500 versions);
- terminal guard start versions: 508–557 (50 versions);
- preferred primary assignment target: 7,500;
- total trainer steps: 558;
- topology: one generation GPU plus one trainer GPU;
- hard caps: six wall-clock hours and 12 GPU-hours.

The qualification data cannot enter the confirmatory estimator. The prior
Qwen3-0.6B power result motivates retaining the accepted information standard,
but equal power on Qwen3-1.7B is not assumed.

## Analysis and stopping

The primary estimator is the accepted cross-fitted generalized-regression
adjustment for opportunity and its zero indicator, with the original unadjusted
estimator reported only for continuity. Inference remains clustered by start
weight version with four-lag Bartlett HAC, eight-version circular blocks, 20,000
bootstrap draws, and the outer interval envelope. Materiality requires the lower
endpoint to exceed `Delta_L=0.2` at the registered alpha.

The 558-step window is fixed. There is no outcome-guided stopping, automatic
extension, rerandomization, or retry. Falling short of the assignment target is
reported; it does not authorize adding steps.

## Gate sequence

Before acquisition, the dedicated parser, detached pipeline, configuration,
protocol, and hashes must pass a containerized no-training preflight. A terminal
green preflight permits an acquisition decision but does not itself authorize
the 558-step run.

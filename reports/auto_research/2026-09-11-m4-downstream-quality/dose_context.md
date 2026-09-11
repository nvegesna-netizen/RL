# Retrospective dose-context audit

## Scope

This descriptive audit uses the eight authenticated Llama 3.2 lifecycle ledgers
already acquired for M4. It does not change a registered M4 estimate, establish
the prevalence of natural production delays, or measure final model quality.
The canonical machine-readable result is `dose_context.json`, SHA-256
`25191cceeb1fc174c554126c6ee33e991b59e9a64f73035e0c5644b85976c062`.

## Results

| Run | Median update (s) | 5 s / update | Median generation (s) | 5 s / generation | d5 overshoot p99 (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Llama 1B / OpenMath r1 | 8.671 | 0.577 | 8.172 | 0.612 | 0.058 |
| Llama 1B / OpenMath r2 | 8.886 | 0.563 | 8.511 | 0.587 | 0.062 |
| Llama 1B / GSM8K r1 | 8.558 | 0.584 | 8.157 | 0.613 | 0.067 |
| Llama 1B / GSM8K r2 | 8.438 | 0.593 | 7.326 | 0.682 | 0.070 |
| Llama 3B / OpenMath r1 | 15.259 | 0.328 | 15.366 | 0.325 | 0.036 |
| Llama 3B / OpenMath r2 | 14.971 | 0.334 | 15.170 | 0.330 | 0.032 |
| Llama 3B / GSM8K r1 | 11.522 | 0.434 | 11.102 | 0.450 | 0.037 |
| Llama 3B / GSM8K r2 | 15.465 | 0.323 | 15.383 | 0.325 | 0.065 |

Across these runs, five seconds is 0.323--0.593 median learner-update intervals
and 0.325--0.682 median single-sibling generation durations. The observed d5
hold median is 5.001--5.002 seconds, and the largest run-level p99 overshoot is
0.070 seconds. The controller interval from the last sibling completion to hold
start is approximately 0.9--2.9 milliseconds at the median.

The earlier authenticated Qwen cadence audit places the same intervention at
0.379--0.644 median learner-update intervals. Together, the two audits show that
the fixed perturbation spans a substantial fraction of one update cycle in every
reported setting and is delivered precisely. They do not show how frequently an
unmodified production scheduler would add a delay of this magnitude.

## Consequence for the downstream study

Retain five seconds for the first run-level outcome study because it preserves a
validated intervention and represents a meaningful within-run cadence
perturbation. Report cadence-normalized dose as context rather than redefining
the treatment after outcomes. The prospective study should additionally report
run-level throughput and time-to-quality so reviewers can interpret the quality
effect against systems cost.

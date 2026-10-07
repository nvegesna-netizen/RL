# M4 gradient-utility audit: terminal report

Status: **MAGNITUDE_ONLY**

The preregistered no-update audit provides positive construct evidence for
gradient magnitude, but not for gradient direction. On the 214 finite groups
with positive M4 and nonzero exact gradients, the baseline model using token
count, reward mean, reward variance, and truncation achieved cross-fitted
\(R^2=0.758\) for log exact gradient norm. Adding M4 increased cross-fitted
\(R^2\) to `0.936`, a gain of `0.179`; its simultaneous 95% interval was
`[0.133, 0.241]`.

The two directional tests did not meet their frozen conditions. Adding M4
changed signed-consensus-utility MSE by `-0.000534` and `-0.000550` under the
two sketch seeds, where positive values denote improvement. Their simultaneous
intervals were `[-0.001586, 0.000544]` and `[-0.001460, 0.000372]`. The M4
minus reward-variance rank differences for positive consensus utility were
`-0.0950` and `-0.1013`, with intervals `[-0.2520, 0.0674]` and
`[-0.2572, 0.0576]`. Thus the data support conditional prediction of update
magnitude, not preferential alignment with the held-out consensus direction.

This distinction is scientifically useful. M4 is not merely an observed-loss
accounting quantity: it carries held-out information about the size of the
actual full-model gradient beyond the frozen service/reward covariates.
However, a large update is not necessarily a useful update. The result does
not justify claiming that an M4-prioritized scheduler improves learning, and
the frozen rule does not authorize a scheduler outcome acquisition.

## Integrity and measurement

All preregistered integrity gates passed: 256 unique measured groups, eight
siblings per group, 214 primary-population groups, 42 zero-M4 groups, 42
zero-gradient groups, acknowledged aborts for every measured group, no call to
`finish_train_step`, zero optimizer and scheduler steps, learner version zero,
and identical pre/post parameter hashes. The runtime generated 320 neutral-arm
groups; 256 were measured and 64 were removed at bounded shutdown.

CountSketch fidelity was well inside the frozen limits. Median relative norm
errors were `0.00355` and `0.00356`; p95 errors were `0.00998` and `0.01009`.
Corrected observer duty was `0.000348` (0.0348%). No prompts, completions, raw
gradients, or checkpoint were retained.

The eight-file runtime inventory reproduced exactly. An independent local
10,000-resample replay returned the same `MAGNITUDE_ONLY` classification and
all numeric values agreed within `1e-12` relative tolerance; the largest
absolute platform-level difference was `7.8e-16`.

## Claim boundary

The supported conclusion is limited to the frozen Llama-3.2-1B/GSM8K
checkpoint and no-update design: M4 adds substantial conditional information
about exact group-gradient magnitude beyond the registered baseline features.
The audit does not show incremental directional utility, terminal-quality
improvement, scheduler efficacy, production readiness, or generalization to
other models or workloads.

Execution identifiers below are provenance only, not scientific endpoints:

- parent `72218730`, child `72218841`;
- main job `474425611`, Slurm job `6191220`;
- main artifact ZIP SHA-256
  `fdadac808517b360d937a94cd3bd287879864172774974e391205fc3e03d49db`;
- logs-after ZIP SHA-256
  `488128812f2cfc6c7672305c2d16fff9ca65151e2b3b2bbbf57c660166d15263`;
- authenticated gradient ledger SHA-256
  `a226a6de068256fca55a48a8be931cd5ae33896cdfa4e92d6c55d13b414124d6`;
- terminal analysis SHA-256
  `d06c02df9047df0f21eab9e62d5d099eb2de434585f6018592141b4e3e0ae3ff`.

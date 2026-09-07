# Qwen3-1.7B GSM8K confirmatory acquisition R2 result

Status: `TERMINAL_COMPLETE_NOT_MATERIAL`.

The prospectively repaired R2 acquisition completed all 558 trainer steps and
produced 9,429 primary-window assignments. Under the preregistered adjusted
estimator, normalized opportunity loss was **0.1377955**, with a 95% confidence
envelope of **[0.1152335, 0.1599156]**. The entire envelope is below the frozen
materiality threshold of 0.20, so the primary causal conclusion is
`NOT_MATERIAL` for Qwen3-1.7B on GSM8K in this tested environment.

This is not a zero-effect conclusion. The estimate is positive, but the design
asked whether it met the prespecified materiality threshold. The supporting
unadjusted estimate was 0.1731261 with envelope [0.0737453, 0.2691661] and was
`INCONCLUSIVE`; by protocol, it cannot override the adjusted primary result.

## Mechanism and portability

The controlled-delay mechanism replicated. Across the primary window, the d5
arm had 4,756 assignments, direct-chain rate 0.3347351, and mean version advance
0.5447855; the control arm had 4,673 assignments and zero for both mechanism
statistics. All mechanism checks passed and no primary assignment was unscored.

Observer duty was supported: corrected duty was 0.0029434 across 10,502
observations, below the 0.01 maximum. Mechanism replication and observer-duty
support are qualifying evidence and do not alter the primary causal conclusion.

## Execution and provenance

Parent pipeline `66570975`, trigger bridge job `428678429`, and downstream
pipeline `66571072` succeeded. Downstream jobs `428679286` (logs-before),
`428679287` (EOS compute), and `428679288` (logs-after) all succeeded. Slurm job
`5986731` exited 0. The compute job ran for 7,608.3 seconds, within the frozen
four-hour wall-clock and eight-GPU-hour caps. No retry or extension occurred.

The run used source commit
`b79e419aea95b5226ac4274c9a1535251658b5a5`, manifest SHA-256
`a7420a4252940cd1cfc6ab48883d1312d3d850fd1602830733b050abea438180`,
and protocol SHA-256
`3413bde3718563157cee5d504f174710f65406dd7f1effd8ad2755a03d73b69e`.
The preserved 134,908,947-byte terminal archive has SHA-256
`966dfbf60548e5fd791d59d3baa2bc3b3a1b8d6b0b1a15de1ef510fc618720ca`;
its ZIP integrity test and all seven embedded checksum-ledger entries passed.
Large raw artifacts remain outside Git.

An independent strict ledger reconciliation joined 158,136 lifecycle rows and
11,061 opportunity rows into 9,429 unique primary assignments, confirmed full
coverage of learner versions 8 through 507, terminal delivery status for every
assignment, 558 train-step rows, and final learner version 558. R1 and
qualification observations did not enter the estimator.

A separate full 20,000-draw local analyzer replay exceeded a 900-second local
execution cap before writing an output file. It produced no conflicting
estimate and is recorded as an incomplete verification attempt, not as a
successful reproduction. The terminal estimate therefore rests on the
preserved analyzer output and verified provenance chain, supplemented by the
completed independent ledger reconciliation above.

## Integrated conclusion

The accepted OpenMath experiment remains `MATERIAL` in its tested setting. This
new, prospectively specified GSM8K transport study is `NOT_MATERIAL`: the delay
mechanism replicated, but the primary adjusted opportunity-loss envelope lies
below 0.20. Together, these results support workload heterogeneity. They do not
justify erasing the OpenMath finding, retroactively treating earlier incomplete
GSM8K attempts as scientific results, or generalizing either result beyond its
tested workload and environment.

Machine-readable evidence is in `acquisition_r2_result.json`.

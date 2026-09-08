# M4 four-cell common-window grid synthesis

Status: `INTERACTION_DETECTED`.

The preregistered secondary synthesis used start versions 8-407 in all four
cells, complete terminal scoring, eight-fold adjusted estimation, four-lag HAC,
eight-version circular blocks, and 20,000 independently seeded bootstrap draws
per cell. Every protocol and raw ledger hash matched its preserved terminal
result before analysis.

The common-window adjusted cell estimates were:

| Cell | Assignments | Estimate | HAC SE |
| --- | ---: | ---: | ---: |
| Qwen3-0.6B / OpenMath | 7,199 | 0.3054232108 | 0.0150455344 |
| Qwen3-0.6B / GSM8K | 7,710 | 0.2332654341 | 0.0118136656 |
| Qwen3-1.7B / OpenMath | 6,955 | 0.2982263079 | 0.0145328316 |
| Qwen3-1.7B / GSM8K | 7,613 | 0.1430635290 | 0.0125264634 |

The OpenMath-minus-GSM8K workload contrast is `0.0721577766` at 0.6B and
`0.1551627789` at 1.7B. The registered difference in those contrasts,

`(0.6B OpenMath - 0.6B GSM8K) - (1.7B OpenMath - 1.7B GSM8K)`,

is `-0.0830050022`. Its combined independent-cell HAC interval is
`[-0.1361068717, -0.0299031327]`; its bootstrap interval is
`[-0.1365688691, -0.0289398546]`. The registered 95% outer envelope is
therefore `[-0.1365688691, -0.0289398546]`, which excludes zero. The secondary
conclusion is `INTERACTION_DETECTED`.

Equivalently, the tested OpenMath-versus-GSM8K M4 opportunity-loss contrast is
smaller at 0.6B than at 1.7B. The within-workload 1.7B-minus-0.6B contrasts are
`-0.0071969029` for OpenMath and `-0.0902019051` for GSM8K. These are contrasts
of the M4 opportunity-loss estimand, not model-quality or task-accuracy effects.

This result is secondary and cannot override any cell's registered primary
conclusion. In particular, the new Qwen3-0.6B/GSM8K primary result remains
`MATERIAL`. The interaction supports heterogeneity across the four tested
model/workload cells; it does not establish family-wide generalization beyond
the two models, two workloads, instruments, runtimes, and environments tested.

The deterministic terminal synthesis receipt has SHA-256
`56fa64ac52355a863c3248e4c9d5c9d3754ac1be0b19ca99bff8fa539587df1c`.
The complete compact evidence map, including protocol, ledger, terminal-result,
bootstrap-shift, runner, and analysis-code hashes, is preserved in
`grid_synthesis_result.json`.

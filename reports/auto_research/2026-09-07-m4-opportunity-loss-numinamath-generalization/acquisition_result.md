# NuminaMath-1.5 paired M4 generalization result

Status: `SAME_DIRECTION_GENERALIZATION`.

The preregistered primary interaction was

`(0.6B NuminaMath - 0.6B GSM8K) - (1.7B NuminaMath - 1.7B GSM8K)`.

Its estimate is `-0.0725802885`. The independent-cell HAC interval is
`[-0.1291134561, -0.0160471210]`, and the 20,000-draw circular-block bootstrap
interval is `[-0.1305496377, -0.0149320766]`. The registered 95% outer envelope
is therefore `[-0.1305496377, -0.0149320766]`, wholly below zero. The primary
conclusion is `SAME_DIRECTION_GENERALIZATION`.

This replicates the direction of the prior OpenMath-versus-GSM8K interaction:
the NuminaMath-minus-GSM8K M4 contrast is larger at 1.7B than at 0.6B. The
contrast is `0.0822154333` at 0.6B and `0.1547957218` at 1.7B. These are
contrasts of normalized M4 opportunity loss, not task accuracy or model quality.

## Supporting cell results

Both prospective NuminaMath cells completed all 448 trainer steps and every
required start version from 8 through 407, with complete terminal scoring.

| Cell | Assignments | Adjusted estimate | 95% outer envelope | Conclusion |
| --- | ---: | ---: | ---: | --- |
| Qwen3-0.6B / NuminaMath | 7,229 | 0.3154808674 | [0.2877998311, 0.3428558739] | `MATERIAL` |
| Qwen3-1.7B / NuminaMath | 7,050 | 0.2978592508 | [0.2614689302, 0.3340215465] | `MATERIAL` |

Each cell exceeded the frozen 6,900-assignment minimum. The mechanism conclusion
is `REPLICATED` in both cells. Corrected observer duty is `0.0021208679` at 0.6B
and `0.0017982187` at 1.7B, both below the registered `0.01` ceiling, so both
observer-duty qualifiers are `SUPPORTED`. These cell results are supporting
endpoints and cannot override the primary paired interaction.

## Acquisition and provenance

The 0.6B parent/downstream pipelines were `66843216`/`66843373`; compute job
`430968079`, logs-after job `430968080`, and Slurm job `5996023` succeeded. The
terminal artifact SHA-256 is
`ad7c556f435b0f9e30a65c201680bf6dab74e077af16e38f299bfc04af969c35`.

The 1.7B parent/downstream pipelines were `66843180`/`66843327`; compute job
`430967786`, logs-after job `430967787`, and Slurm job `5996009` succeeded. The
terminal artifact SHA-256 is
`6aa61968730833387a15b45a6d0f2be541c6f84b6d4f16a29e7dbf583981fc9b`.

Both 39-member archives passed ZIP integrity and their embedded checksum
ledgers. Their capture summaries bind source commit
`02b0ad4abfedc80994f71f8a550473431154b823`, protocol SHA-256
`bbe2c07211952e3e46d4511524f68e864606a7c63f271ae8e064263044dad0c8`,
and the two exact model configs. Both packages were submitted before either
outcome was inspected. Qualification observations did not enter an estimator;
no retry or extension occurred.

Two full local executions produced byte-identical terminal result SHA-256
`0dfd265d788d6697cbfcabc1f39519b23191b8196279894de66f98aaee8af018`.
The immutable historical GSM8K evidence record used by the analysis has SHA-256
`4a33668f332e250c71769ecc88b57972ecd08c6f5c69b38d31e53fa93240168f`.

## Bounded conclusion

The evidence supports same-direction generalization of the previously detected
model-by-workload M4 interaction from OpenMath-versus-GSM8K to
NuminaMath-versus-GSM8K in the two tested Qwen3 model scales. It also supports a
material M4 opportunity-loss effect in each tested NuminaMath cell. It does not
establish family-wide generalization to other models, workloads, algorithms,
runtimes, hardware, or deployment environments.

# Qwen3-1.7B M4 confirmatory transport result

Status: terminal `MATERIAL` result in the tested setting. The evidence supports a
material M4 opportunity-loss effect for Qwen3-1.7B under the tested
OpenMathInstruct-2, GRPO, two-H100 EOS configuration. This does not establish
generalization beyond that setting.

## Registered primary result

The adjusted control-versus-five-second-delay estimate is
`0.3024933052610129`, relative to the frozen materiality threshold of `0.2`.
The 95% outer confidence envelope is
`[0.27688057033389485, 0.328106040188131]`; its lower endpoint exceeds the
threshold. The registered material p-value is `4.999750012499375e-05`, so the
prospective primary conclusion is `MATERIAL`.

All 8,673 primary assignments were scored: 4,376 control and 4,297 d5. The
coverage gate passed and no missingness identification width remained. The
supporting unadjusted estimate was `0.2793816078235677` with envelope
`[0.21403546095640158, 0.3447277546907338]`; it is also material but cannot
override the registered adjusted primary endpoint.

## Mechanism and observer checks

The registered mechanism conclusion is `REPLICATED`. Control had zero direct
chains and zero mean version advance. D5 had a direct-chain rate of
`0.2229462415638818` and mean version advance of `0.35978589713753784`. Every
mechanism check passed and no primary assignment was unscored.

The portability/measurement qualifier is `SUPPORTED`. Corrected observer duty
was `0.0018525392987851206` (0.1853%) across 9,676 observations, below the
registered 1% ceiling.

## Acquisition and provenance

The successful recovery used the unchanged frozen scientific protocol and a
four-hour EOS scheduler limit:

- source commit: `a1c719177704b7d14cf1d503dc96f0f2f9f1701d`;
- source archive SHA-256:
  `2c8af8d67dd85d49f70bf71a58d3eb8e90e2cf56915eb8e1ca398e34024de2cf`;
- protocol SHA-256:
  `eef87a5e26f2cc1a39428628b0c28288b17a297f49d449a80ed9699f9ee17cb7`;
- recovery manifest SHA-256:
  `0a865cafb6c25a29b03521c790e6d5d78a01045ccdfaffa3c1b1057198ca633d`;
- parent pipeline `66186545`, downstream pipeline `66186620`, EOS compute job
  `425431228`, and artifact job `425431229`;
- EOS compute duration: `8709.859687` seconds;
- terminal raw artifact: 131,003,667 bytes, SHA-256
  `022e4d18f2eedd2339c3c7eb7decf6e21f929761de46bcd0f85a73f0a2700ac7`.

The raw artifact remains outside Git. Every terminal member matches the embedded
hash ledger. The run log contains the exact 558-step/558-version completion
marker, and the acquisition preflight records 135 passing tests. Compact member
hashes and all reported numbers are preserved in `acquisition_result.json`.

## Historical transport failure

The first acquisition pipeline, `66184233`, failed before allocation because
EOS `batch` rejected a six-hour Slurm request. It created no lifecycle,
opportunity, observer-duty, or result data and therefore is not a scientific
acquisition. The recovery changed only the scheduler cap to EOS's four-hour
maximum; it did not change the protocol, source, arms, estimator, assignments,
or fixed 558-step window.

## Interpretation

This is a successful one-axis model-scale transport test of the accepted M4
instrument. It extends the material result from the prior Qwen3-0.6B tested
setting to this Qwen3-1.7B tested setting. It does not support claims about other
model families, workloads, algorithms, hardware, or environments.

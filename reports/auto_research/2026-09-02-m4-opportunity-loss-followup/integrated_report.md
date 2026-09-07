# Integrated M4 opportunity-loss study report

Status: complete for the original tested setting, with subsequent one-axis
transport evidence incorporated. The evidence supports a material M4
opportunity-loss effect on OpenMath in the tested Qwen3-0.6B and Qwen3-1.7B
settings, but the prospectively registered Qwen3-1.7B/GSM8K result is
`NOT_MATERIAL`. The accepted instrument and mechanism findings remain valid.

## Final conclusion

The registered prospective control:d5 follow-up estimated adjusted opportunity
loss of `0.3054232108`. Its 95% confidence envelope,
`[0.2759345053, 0.3349119163]`, lies entirely above the registered `0.20`
materiality threshold. The materiality p-value is `0.0000499975001`.
The mechanism result is `REPLICATED`, and the portability qualifier is
`SUPPORTED`.

This conclusion is limited to the model, workload, runtime, and instrumentation
used in these acquisitions. It does not establish generalization to other models,
workloads, or deployment environments.

## Evidence sequence

> validated instrument → replicated mechanism → imprecise first materiality
> result → prospectively redesigned follow-up → material effect confirmed

### 1. Validated M4 instrument

The accepted instrument combined the common opportunity ledger, terminal
missingness bounds, detached protocol-bound inference, corrected observer-duty
measurement, and direct-chain/version-advance mechanism analysis. The validation
chain was implemented through commits `c0d12e61f`, `5940059c8`, `766351123`,
`4e6f6a993`, and `9cc2e9c6e`; the first acquisition froze source commit
`9cc2e9c6e339f7cd3c0e83fa9abf80af9cf428a6`.

“Successful instrument” means that the measurement and verification contracts
worked and produced complete, protocol-bound evidence. It is distinct from the
statistical decision on the primary materiality endpoint.

### 2. First confirmatory acquisition

The 224-step control:d5:d10 acquisition ran in parent pipeline
[65911886](https://gitlab-master.nvidia.com/dl/jet/ci/-/pipelines/65911886),
downstream pipeline `65912780`, and EOS job `423068757`. It scored all 3,298
primary assignments: 1,351 control, 1,379 d5, and 568 d10, with none unscored.

The mechanism result was `REPLICATED`. Direct-chain rates were `0` for control,
`0.2668600435` for d5, and `0.5862676056` for d10; mean version advances were
`0`, `0.4372733865`, and `0.8045774648`. Corrected observer duty was
`0.0023149549`, below the `0.01` cap, so portability was `SUPPORTED`.

The primary opportunity-loss estimate was positive at `0.2098068292`, but its
95% confidence envelope was `[0.1159593136, 0.3015376009]`, crossing the `0.20`
threshold. Its registered materiality p-value was `0.4218789061`. Therefore this
specific primary endpoint remains `INCONCLUSIVE`; it was neither evidence that
the instrument failed nor a negative materiality finding.

The parent pipeline was red only after the canonical result had been produced,
because summary packaging raised a syntax error. The preserved main artifact has
SHA-256 `e81a29747628b6f0490323207f317755de411d0f8e7527f715368919aee12fb7`,
and the canonical result has SHA-256
`6c3fc0cf0567d4dab63153769c075dc7d491e41cb2ceb982b569649f5b42ced3`.

### 3. Prospective redesign

The completed artifact was used only as exploratory design input. Cross-fitted
pre-delay opportunity adjustment suggested stable efficiency gains. The formal
estimator was implemented at `7403f11fb`, and its bootstrap, coverage, and sharp
missingness contract was frozen at `1bf0eb896`.

The new design removed the historical d10 positive-control arm, allocated
control:d5 at 1:1, and specified 448 trainer steps. A 20,000-draw empirical
version-cluster simulation estimated power `0.81555` at the prospective
alternative `0.25`, exceeding the `0.80` design gate. This design exercise did
not reclassify the first acquisition.

### 4. Frozen follow-up preflight

The follow-up protocol was implemented at `3681e528b` and frozen at source commit
`47dc1a713cbf0e07813d31c4ba70a8262214e098`. The no-training preflight completed
successfully in parent pipeline
[65942089](https://gitlab-master.nvidia.com/dl/jet/ci/-/pipelines/65942089),
downstream pipeline `65942720`, and compute job `423347400`.

All 111 selected tests passed, configuration and hashes resolved, and the job
reported both training and acquisition as not started. The preflight artifact
SHA-256 is `544bff5bd9ac5799b1890ab4807e598caea8a78c9bcda72bb549234d94852968`.
The frozen protocol SHA-256 is
`6631a4ca8cfc6010150d1c135a1f5821c717f0dea189a3309c5f0f58f9b0709c`.

### 5. Prospective follow-up result

The registered acquisition completed successfully in parent pipeline
[65985843](https://gitlab-master.nvidia.com/dl/jet/ci/-/pipelines/65985843),
downstream pipeline `65986159`, and EOS compute job `423739790`. It completed all
448 trainer steps and scored all 7,199 primary assignments: 3,520 control and
3,679 d5.

The adjusted primary result was:

- estimate: `0.3054232108`;
- HAC standard error: `0.0150455344`;
- 95% confidence envelope: `[0.2759345053, 0.3349119163]`;
- circular-block bootstrap interval: `[0.2774491724, 0.3344227273]`;
- materiality p-value: `0.0000499975001`;
- conclusion: `MATERIAL`.

The supporting unadjusted estimate was `0.3335223005`, with confidence envelope
`[0.2635293811, 0.4014294320]`. It agreed with, but cannot override, the adjusted
registered primary analysis.

The mechanism again `REPLICATED`: the direct-chain rate was `0/3520` in control
and `891/3679 = 0.2421853765` in d5, and the d5-control mean version-advance
contrast was `0.4052731721`. Corrected observer duty was `0.0021738545`, below
the `0.01` cap, so portability was `SUPPORTED`.

The preserved 54,950,561-byte artifact has SHA-256
`07cddc489ea20055161f9c435d44d0a52b91113fda9eac16472bdb531dc6ffb6`.
Its canonical result member has SHA-256
`01a5867753462c3d956c0c44f730ca6ae4d40859a5e580606544dba70c2c4344`.
The compressed transport changed only packaging; it retained source commit
`47dc1a713cbf0e07813d31c4ba70a8262214e098` and the frozen protocol hash above.
No automatic retry or outcome-guided extension occurred.

## Decision record

The first primary materiality result remains `INCONCLUSIVE` as a historical fact.
The accepted M4 instrument and both mechanism results remain successful. The new
prospective acquisition is an independent registered follow-up and supplies the
terminal `MATERIAL` conclusion. Nothing is retroactively relabeled or erased.

The precise supported claim is:

> The evidence supports a material M4 opportunity-loss effect in the tested
> setting.

The evidence supports model-scale transport from Qwen3-0.6B to Qwen3-1.7B on
OpenMath. It does not support workload transport from OpenMath to GSM8K at the
registered 0.20 threshold for Qwen3-1.7B.

## Subsequent transport evidence

The one-axis Qwen3-1.7B/OpenMath transport acquisition estimated adjusted
opportunity loss of `0.3024933053`, with 95% envelope
`[0.2768805703, 0.3281060402]`. It completed 558 steps and 8,673 primary
assignments, replicated the mechanism, and supported observer duty. Its terminal
conclusion is `MATERIAL` in that tested setting.

The next one-axis study retained Qwen3-1.7B and changed the workload to GSM8K.
Its prospective R2 acquisition completed 558 steps and 9,429 primary
assignments. The adjusted estimate was `0.1377955456`, with 95% envelope
`[0.1152334598, 0.1599156329]`, entirely below `0.20`; its conclusion is
`NOT_MATERIAL`. The mechanism still `REPLICATED` and observer duty remained
`SUPPORTED`, so this is a positive but sub-threshold workload result rather than
an instrument failure or a zero-effect claim.

The OpenMath model-scale difference is only `-0.0029299055` when expressed as
Qwen3-1.7B minus Qwen3-0.6B. At Qwen3-1.7B, the OpenMath-minus-GSM8K difference
is `0.1646977597`. These completed results support workload heterogeneity, but a
formal model-by-workload interaction is not established: the Qwen3-0.6B/GSM8K
cell is missing and the existing primary windows are not all identical.

## Provenance

Machine-readable provenance and exact values are in
[`evidence_map.json`](evidence_map.json). The prospective design is documented in
[`prospective-efficiency-audit/design.md`](prospective-efficiency-audit/design.md),
and its terminal result is recorded in
[`prospective-efficiency-audit/result.json`](prospective-efficiency-audit/result.json).
Large raw artifacts remain outside Git and are bound by the SHA-256 values above.

## Recommended next scientific direction

Do not repeat any completed acquisition. The smallest informative successor is a
separately preregistered Qwen3-0.6B/GSM8K cell, which would complete the existing
two-model-by-two-workload grid while changing only model scale relative to the
completed GSM8K study. Before acquisition, an offline harmonization audit must
fix a common-window secondary analysis, effect threshold, sample size, exclusion
rules, and interaction-inference contract. This report authorizes no launch.

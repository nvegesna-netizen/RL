# M4 downstream-quality terminal result

Status: `COMPLETE_PRIMARY_INCONCLUSIVE`

## Evidence lock

The terminal population contains all 16 prospectively registered matched
training-seed blocks and all 32 registered runs. Eighteen runs use their
original successful workload artifacts; the 14 b10--b16 identities use the
separately authorized runtime-recovery-v2 artifacts. Every selected scientific
workload succeeded. All 32 workload ZIPs passed integrity checks, all signed
artifact manifests and registered members matched their declared SHA-256, and
all terminal results contain the registered 448/448 endpoint and exactly 1,024
binary prompt scores in the same prompt order.

The completion gate is `COMPLETE_AUTHENTICATED_16_PAIRS`. Total measured
runtime is 52.1158 wall-hours and 104.2317 H100 GPU-hours, below the frozen
128-wall-hour and 256-H100-hour caps. The b02 immediate parent remains marked
failed solely because its logs-after transfer failed; its scientific workload
succeeded and its workload archive is authenticated.

The runtime recovery itself is operationally validated: all 14 repaired jobs
rendered exactly one `--time 0:14400`, retained `--deadline now+8hours`, and
completed with Slurm state `COMPLETED` and exit `0:0`.

## Registered primary result

The estimand is terminal OpenMath accuracy under mixed-d5 release minus
all-immediate release, with the matched training-seed block as the independent
unit.

- mixed-d5 mean accuracy: 0.21484;
- all-immediate mean accuracy: 0.25977;
- paired mean difference: -0.04492 (-4.49 percentage points);
- paired sample SD: 0.16706;
- standard error: 0.04177;
- two-sided paired Student 95% interval: [-0.13394, 0.04410];
- registered 0.02-margin conclusion: `INCONCLUSIVE`.

The block contrasts are highly heterogeneous: ten are negative, five positive,
and one zero, ranging from -0.32617 to +0.31250. The median contrast is
-0.01367. Two immediate endpoints and four mixed-d5 endpoints have zero
accuracy; this is descriptive evidence of substantial run instability, not a
post hoc exclusion rule.

## Frozen sensitivities

The exact 65,536-assignment within-block sign-flip test gives two-sided
`p = 0.29816`. The 20,000-draw paired bootstrap percentile interval is
[-0.12402, 0.03442], and its studentized interval is
[-0.13185, 0.04598]. The prompt-fixed-effect sensitivity has the same point
estimate and cluster-robust SE 0.04244. Balanced prompt adjustment using V8
base correctness leaves the point estimate unchanged. Conclusions remain
`INCONCLUSIVE` at absolute materiality margins 0.01, 0.015, 0.025, and 0.03.

## Scientific interpretation

The point estimate is in the hypothesized harmful direction, but the interval
includes material harm, no effect, and benefit. The study therefore neither
confirms a final-quality cost nor establishes practical equivalence. It does
show that final-quality effects are much noisier across independent training
runs than the proximal opportunity-loss mechanism.

For the paper, this is useful as a prospective end-to-end test and an honest
scope boundary. It does not support expanding the headline claim beyond causal
opportunity loss in the tested systems. Any successor would be a new hypothesis
test with a stability intervention and power recalibrated to the observed
run-level variance, not a retry, replacement, exclusion of zero endpoints, or
extension of this frozen experiment.

No further run is authorized by this result.

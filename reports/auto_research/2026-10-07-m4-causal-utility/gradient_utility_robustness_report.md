# M4 gradient-utility post-primary robustness report

## Scope

This report stress-tests the frozen magnitude interpretation from the completed
256-group, no-update gradient-utility acquisition. These analyses were run only
after the registered terminal analysis had classified the acquisition as
`MAGNITUDE_ONLY`; they are descriptive robustness checks and do not change that
classification.

The authenticated gradient ledger has SHA-256
`a226a6de068256fca55a48a8be931cd5ae33896cdfa4e92d6c55d13b414124d6`.
The committed compact terminal result binds the acquisition protocol, source,
artifact, and replay provenance. Large raw artifacts remain outside Git.

## Primary reconstruction and interpretation

The primary population contains 214 groups with positive M4 opportunity and a
positive exact gradient norm. The frozen eight-fold analysis is reproduced
exactly:

| Model | Cross-fitted R² |
| --- | ---: |
| Baseline covariates | 0.757642 |
| Baseline covariates + log M4 | 0.936437 |
| Increment from adding M4 | +0.178795 |
| M4 alone | -0.006450 |

The registered simultaneous interval for the incremental R² is
[0.133276, 0.240928]. M4 therefore contains substantial information about
exact gradient magnitude *conditional on* valid actor tokens, reward mean,
reward variance, and truncation. It is not an effective standalone magnitude
predictor in this acquisition. This conditional interpretation is more precise
than describing M4 as a generic gradient-magnitude proxy.

## Stress tests

Every tested specification retained a positive incremental R²:

| Check | Result |
| --- | --- |
| Leave one frozen fold out | +0.167513 to +0.194364 across eight omissions |
| 100 alternative balanced fold assignments | median +0.173787; range +0.166885 to +0.184653 |
| 1% / 5% winsorization | +0.184239 / +0.151937 |
| Include all 256 groups, including zero M4 and zero gradient | +0.054824 |
| Fixed ridge alpha from 1e-4 through 100 | +0.042618 to +0.178567 |

A 1,000-draw within-frozen-fold permutation check preserved the ordinary
covariates and shuffled only M4. None of the permuted gains equaled the observed
+0.178795 gain (finite-sample one-sided p=0.000999). The permuted median was
-0.001644, the 95th percentile was +0.002076, and the maximum was +0.008692.
This rejects an explanation based only on the frozen fold structure or an
arbitrary extra regressor.

## Directional boundary

The two frozen directional seeds did not show improvement. For seed 20261019,
the directional MSE gain was -0.000534 with simultaneous interval
[-0.001586, 0.000544], and the signed-rank difference was -0.09499
[-0.25195, 0.06738]. Seed 20261021 produced the same conclusion: MSE gain
-0.000550 [-0.001460, 0.000372] and signed-rank difference -0.10127
[-0.25725, 0.05765].

The supported result is consequently narrow: M4 contributes robust,
out-of-fold information about gradient *magnitude* beyond the registered
baseline covariates. The study does not show that M4 predicts gradient
direction, works alone as a ranker, improves a scheduler, improves terminal
quality, or generalizes outside this acquisition.

## Reproduction

Run:

```text
python analyze_gradient_utility_robustness.py \
  --ledger /path/to/m4-gradient-utility-acquisition-gradient.jsonl \
  --output gradient_utility_robustness_result.json
```

The analyzer SHA-256 is
`274a46fa8a323530270a67c390698e3c278f0fb1067fdc2078802f6f33f6a8f0`.
The generated result SHA-256 is
`9450ccd7f2e4d6a4f02029133dce4053c14cbd03fd668cd7f621fb8e7ea2be9c`.

# Update-cadence sensitivity

This retrospective descriptive analysis places the fixed five-second
treatment on the observed learner-update time scale. It does not redefine
the treatment or any registered causal estimand.

| Model | Workload | Median update (s) | IQR (s) | 5 s / median update |
| --- | --- | ---: | ---: | ---: |
| Qwen3-0.6B | OpenMath | 11.878 | [9.481, 14.169] | 0.421 |
| Qwen3-1.7B | OpenMath | 13.131 | [10.721, 15.764] | 0.381 |
| Qwen3-0.6B | GSM8K | 7.769 | [6.105, 9.867] | 0.644 |
| Qwen3-1.7B | GSM8K | 8.769 | [7.112, 10.722] | 0.570 |
| Qwen3-0.6B | NuminaMath | 12.009 | [9.902, 14.273] | 0.416 |
| Qwen3-1.7B | NuminaMath | 13.201 | [10.569, 16.231] | 0.379 |

Each cell contributes exactly 400 authenticated inter-update intervals:
the time from learner version `v` to `v+1` for preceding versions 8--407.
The ratios are context descriptors only. Using them to rescale the effect
after observing outcomes would change the estimand and is not done.

Result SHA-256: `dd9f0fe6afcc1ca876a24777afd0b196ac479396acc96f45e5f84059bb9ef746`

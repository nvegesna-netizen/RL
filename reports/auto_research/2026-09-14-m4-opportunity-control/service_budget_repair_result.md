# Baseline-budgeted OARS repair result

## Terminal decision

The frozen service-budget repair gate is **PASS**. This authorizes local
shadow-mode implementation and focused non-training tests. It does not authorize
training, EOS submission, qualification, or a prospective acquisition.

All conditions passed without changing the original thresholds:

- the parent authentication, metric-robustness, and natural-relevance conditions
  remain passed;
- repaired shadow L1 gain exceeded 10% in 14/14 acquisitions, versus the frozen
  requirement of 8/14;
- aggregate and every individual decision remained at or below 1.02 times the
  baseline valid-token count;
- the baseline batch was feasible and exactly four groups were selected at all
  1,447 reconstructed contended decisions.

## Result

| Quantity | Result |
| --- | ---: |
| Acquisition-level L1 gain range | 19.3%–71.7% |
| Mean acquisition-level L1 gain | 40.8% |
| Acquisition-level L2 gain range | 21.3%–74.3% |
| Mean acquisition-level L2 gain | 43.4% |
| Aggregate token-ratio range | 0.8506–0.9884 |
| Mean aggregate token ratio | 0.9355 |
| Maximum ratio at any individual decision | 1.019879 |
| Candidate combinations evaluated | 428,726 |
| Maximum candidate-set size | 25 groups |

| Acquisition | L1 gain | L2 gain | Token ratio | Maximum decision ratio |
| --- | ---: | ---: | ---: | ---: |
| Llama 1B · GSM8K · r1 | 45.4% | 51.5% | 0.9861 | 1.0198 |
| Llama 1B · GSM8K · r2 | 56.8% | 55.7% | 0.9750 | 1.0199 |
| Llama 1B · OpenMath · r1 | 57.1% | 53.9% | 0.9718 | 1.0154 |
| Llama 1B · OpenMath · r2 | 46.3% | 45.3% | 0.9884 | 1.0124 |
| Llama 3B · GSM8K · r1 | 46.5% | 62.9% | 0.9160 | 1.0165 |
| Llama 3B · GSM8K · r2 | 71.7% | 74.3% | 0.9880 | 1.0199 |
| Llama 3B · OpenMath · r1 | 25.2% | 30.3% | 0.9715 | 1.0168 |
| Llama 3B · OpenMath · r2 | 29.3% | 30.7% | 0.9620 | 1.0124 |
| Qwen 0.6B · GSM8K | 23.2% | 27.1% | 0.9095 | 1.0181 |
| Qwen 0.6B · NuminaMath | 31.9% | 33.9% | 0.8787 | 1.0018 |
| Qwen 0.6B · OpenMath | 28.0% | 29.6% | 0.8926 | 1.0172 |
| Qwen 1.7B · GSM8K | 19.3% | 21.3% | 0.8506 | 1.0000 |
| Qwen 1.7B · NuminaMath | 54.1% | 54.2% | 0.8893 | 1.0040 |
| Qwen 1.7B · OpenMath | 36.1% | 36.4% | 0.9176 | 1.0023 |

## Interpretation

The repair answers the first gate failure constructively. Opportunity-aware
selection does not need to trade higher coefficient opportunity for more learner
tokens in these reconstructed choice sets. An explicit feasible-set constraint
can preserve the service contract while retaining substantial one-step headroom.

The result also supports using L1 for the first controller: L2 opportunity moved
in the same direction and by a similar or larger amount in every acquisition.
The exact four-group search is modest at the observed choice-set sizes, but this
offline count is not a runtime-latency benchmark.

## Claim boundary

This result is a retrospective policy-development screen. Its choice sets and
future arrivals were produced by the historical policy and the mixed control/d5
experiment. It does not estimate:

- the queue trajectory under budgeted OARS;
- a total causal policy effect;
- learner throughput or wall-clock improvement;
- optimization stability;
- final or learning-curve quality.

Changing selected groups changes all of those downstream quantities. They must
be tested prospectively with whole-run randomization if shadow-mode verification
succeeds.

## Provenance

- Parent result SHA-256:
  `ade3b4ccecdc5920f9ab6e21ec84a4c59538b1956042f66362354cb67e744a96`
- Repair protocol SHA-256:
  `8abf4168a17e08ff51395479829efa2a7754f1899db02491e47e0c5e492d381d`
- Repair analyzer SHA-256:
  `59aebda9f01ed2679ed07ff9eccd17d86eabfe06dbb8375295305b5378ddfe36`
- Repair result SHA-256:
  `018ba2d69b20938cff49966575639a209ebf01e03461b10022bdd1cedfa71d5a`

## Next boundary

Implement baseline-budgeted OARS behind a default-off shadow-mode policy. The
implementation must expose the baseline and proposed identities, scores, token
budgets, and decision latency without changing selection. It must be exactly
equivalent to the existing sampler whenever disabled or uncontended. No training
launch is justified until those contracts and microbenchmarks pass.

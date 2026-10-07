# M4 causal-risk offline transport result

Date: 2026-10-07

Status: `PASS_AUTHORIZE_GRADIENT_UTILITY_AUDIT`

## Result

The frozen offline gate passed. The analysis authenticated and joined all 14
acquisitions and 106,653 registered assignments, used only information known
when the release hold began, and excluded every held-out partition from model
fitting, imputation, and standardization.

Across the 14 leave-one-acquisition-out tests, multiplying the transported
delay-risk estimate by M4 captured an average 0.5317 of observed lost M4 in the
top 10% of assignments. Multiplying the same risk by reward variance captured
0.4866. The registered equal-acquisition mean difference was +0.0451. Every
acquisition-level difference was positive, ranging from +0.0045 to +0.1477.

Both registered family transports were positive:

| Training family | Held-out family | Reward-variance-at-risk capture | M4-at-risk capture | Difference |
|---|---|---:|---:|---:|
| Qwen | Llama | 0.5169 | 0.5751 | +0.0582 |
| Llama | Qwen | 0.3155 | 0.3472 | +0.0318 |

The workload holdouts were also positive: +0.1009 for OpenMath, +0.0971 for
GSM8K, and +0.0123 for NuminaMath.

The result reproduced byte-for-byte. The result SHA-256 is
`049cf55a57f6992c47186183293f678295bd28e08eb5309cfbfd2289228e8bda`.

## What the gate establishes

The pre-release M4 value can be combined with an out-of-domain delay-risk
model to find more of the registered Q-times-non-delivery endpoint under a
fixed selection budget than reward variance can. The gain is not confined to
one family, workload, or acquisition. This supports the operational premise
that pre-consumption value and expiry risk should be represented separately.

Adding M4 to the non-delivery predictor itself made only a small difference.
Across leave-one-acquisition-out folds, mean log-loss changed by -0.00041,
Brier score by -0.000082, and AUROC by +0.00013. Most of the ranking gain
therefore comes from valuing the consequence of non-delivery, not from M4
materially predicting whether non-delivery occurs.

## Important negative diagnostic

The secondary inverse-probability-weighted causal pseudo-outcome did not favor
M4-at-risk in the two family transports. Relative to reward variance at risk,
its top-decile mean was lower by 124.50 when trained on Qwen and tested on
Llama, and lower by 514.01 in the reverse direction. OpenMath was +36.94, while
GSM8K and NuminaMath were -170.28 and -268.98.

These secondary values are noisy ranking diagnostics rather than registered
effect estimates, but they expose the central unresolved issue: finding large
amounts of observed lost M4 is not the same as identifying assignments whose
delay causally destroys learning value. A scheduler efficacy study would be
premature.

## Decision and next gate

The authorized independent gradient-utility audit is now terminal. Registered
M4 added substantial held-out information about exact gradient magnitude
beyond reward variance, token count, reward mean, and truncation: cross-fitted
$R^2$ rose from 0.758 to 0.936, a gain of 0.179 with simultaneous 95% interval
0.133--0.241. The preregistered directional endpoints did not pass under
either sketch seed. The terminal classification is therefore
`MAGNITUDE_ONLY`, which strengthens M4's measurement interpretation but does
not authorize an EOS quality acquisition under the frozen rule.

The audit used a new bounded no-training collector because the prior terminal
archives did not contain replayable token sequences, per-group gradients, or
versioned model checkpoints. It retained exact gradient norms and compact
sketches, not raw gradients, prompts, completions, or a checkpoint.

## Claim boundary

Observed-lost-M4 capture is partly linked by construction to the registered
M4 value in both the score and target. It demonstrates transport of an
operational ranking objective, not independent construct validity. This result
must not be described as terminal-quality improvement, causal mediation,
production readiness, or proof that M4 is marginal learning utility.

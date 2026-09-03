# Qwen3-1.7B neutral resource qualification result

Status: `QUALIFICATION_GREEN`; proceed to a separately frozen prospective
confirmatory design. No confirmatory acquisition is authorized by this result.

The exactly-once neutral qualification completed all 32 trainer steps on the
smallest candidate topology: one Qwen3-1.7B generation GPU plus one Qwen3-1.7B
trainer GPU. Parent pipeline `66124355`, downstream pipeline `66124571`, and EOS
compute job `424883273` are terminal green.

The run produced 544 complete opportunity groups across 32 observed start
versions, or 17.0 groups per observed version. Excluding the high-throughput
initial version and partial terminal version gives 493 groups across 30 interior
versions, or 16.4333 per version. All release assignments used the sole
`neutral` arm with exactly zero delay.

The measured workload wall time was 984.588 seconds. The 32 reported trainer
steps totalled 436.00 seconds, with mean 13.625, median 12.62, and maximum 34.10
seconds; initialization, shutdown, and other unaccounted fixed work totalled
548.588 seconds. Corrected observer duty was 0.00171288 (0.1713%), below the
registered 0.01 ceiling, so observer portability is `SUPPORTED`.

This qualification produces no control-versus-d5 estimate and no causal or
materiality conclusion. Its data are excluded from any later confirmatory
estimator.

## Prospective design gate

The resource gate passes. For planning, use 15 assignments per primary version,
which is deliberately below both the 17.0 whole-window rate and 16.4333 interior
rate. A fixed 500-version primary window therefore plans exactly 7,500
assignments. Preserve the prior geometry with eight burn-in versions and add a
50-version terminal guard:

- burn-in versions: 0–7;
- primary versions: 8–507;
- terminal guard versions: 508–557;
- total trainer steps: 558.

The qualification implies a wall-time planning band of approximately 2.27–4.77
hours and 4.53–9.54 GPU-hours. A prospective hard cap of six wall-clock hours
and 12 GPU-hours is proportionate and leaves no authority for automatic retry or
extension.

Before any acquisition, the control:d5 protocol, estimator, assignment seed and
domain, output paths, source archive, preflight lock, manifest, and one-shot
authority must be frozen separately. The accepted M4 scientific semantics must
remain unchanged.

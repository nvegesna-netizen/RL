# OARS induced-contention systems-preflight result

## Result

The prospectively repaired preflight **passed every frozen systems gate**. This
qualifies the observation-only OARS implementation for the design of a separate
randomized FIFO-versus-OARS experiment. It does not estimate an OARS effect and
does not support a training-quality claim.

The distinction matters: pipeline completion established only that the workload
executed. The scientific systems endpoint came from the authenticated, frozen
result and an independent audit of its ledgers.

## Frozen-gate evidence

- All 64 learner steps completed and produced exactly 64 shadow decisions.
- Every decision contained eight candidate groups and 70 possible four-group
  combinations; all 64 decisions therefore provided the preregistered contention.
- Actual selection matched weight-FIFO in every decision. OARS remained
  observation-only and never selected a group.
- All proposed sets contained four unique groups, remained within the 1.02
  baseline-token service budget, and produced no shadow skips.
- Contended metadata coverage was 100% (64/64 complete decisions).
- Gradient-observer duty was 0.0535%; OARS-observer duty was 0.00161%.
- OARS p95 decision latency was 135,132 ns, or 0.00256% of the median learner-step
  interval. Runtime was 577.19 seconds. Every frozen ceiling passed.

## Provenance

The source commit is `6d5afa213613367cb296846d2a66cc0cdbb1f39e` and
the frozen result SHA-256 is
`e5a33e2297ef364192264f8f2cf108818f1d040b414c9fb165cf4def82a68822`.
The main and logs-after archives passed ZIP integrity; their copies of all six
declared scientific artifacts plus the SHA-256 manifest are byte-identical. The
internal manifest has exact coverage and every hash matches. Large raw archives
remain outside Git and are referenced in the terminal JSON receipt by SHA-256.

## Claim boundary and next gate

The earlier eager-FIFO preflight remains valid evidence that natural cadence
offered no contention in that run. This repair deliberately introduced an
eight-candidate watermark, so it cannot be described as representative of eager
FIFO behavior.

The only unlocked next step is to preregister a separate randomized comparison
of FIFO versus acting OARS, using the same eight-candidate cadence and service
budget in both arms. That experiment requires a new protocol and explicit launch
authorization; it is not a retry or automatic extension of this preflight.

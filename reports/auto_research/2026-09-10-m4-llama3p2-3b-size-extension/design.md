# Prospective Llama 3.2 3B M4 size extension

Status: `TERMINAL_JOINT_MATERIALITY_NOT_CONFIRMED`.

## Purpose and separation

The completed Llama 3.2 1B V5 study remains terminal. This successor asks
whether its material M4 result replicates at 3B within the same named model
family. The 1B ledgers are immutable prespecified references; no 1B acquisition
will be retried, extended, or reclassified.

The design changes model size and nothing else scientifically: OpenMath and
GSM8K, control versus an added five-second delay, lifecycle-derived opportunity,
the 0.20 materiality threshold, versions 8--407, sharp missingness endpoints,
adjustment, HAC, and block bootstrap all remain aligned with V5.

## Scientific decisions

There are two co-primary 3B workload endpoints, each formed from two independent
acquisitions with equal replicate weight. The extension succeeds only if both
OpenMath and GSM8K conclude `MATERIAL` at one-sided alpha 0.05. Holm-adjusted
p-values are also reported for separate workload claims.

The 3B-minus-1B contrasts are secondary. They retain uncertainty from both 1B
and 3B replicates; the completed 1B estimates are not treated as constants.
These fixed-configuration contrasts cannot establish a general scaling law or
a causal effect of parameter count.

## Why qualification is required

The repository contains a Llama 3.2 3B model path, but no 3B asynchronous GRPO
run under the accepted two-GPU M4 topology. The 1B runtime therefore cannot
qualify 3B memory, throughput, model access, or the four-hour limit.

The sequence has two outcome-blind operational gates:

1. One no-training preflight may validate frozen hashes, configuration
   resolution, model/tokenizer metadata access, imports, and topology. It may
   not load model weights or construct a trainer.
2. If separately authorized after a green preflight, paired neutral 64-step
   qualifications may measure capacity on OpenMath and GSM8K. Both must be
   submitted before either is inspected, and neither may enter a causal
   estimator.

## Qualification gates

Each neutral workload must pass every gate:

- exactly 64 completed trainer transitions;
- complete lifecycle-derived opportunity over versions 8--55, with no empty
  version and no imputation;
- one-sided 95% block-bootstrap lower projection of at least 5,000 assignments
  for one 400-version replicate;
- one-sided 95% projected total runtime at most 12,600 seconds, leaving 1,800
  seconds below the 14,400-second scheduler cap;
- corrected observer duty at most 0.01, binary reward support, valid topology,
  and authenticated output hashes.

The assignment threshold is lower than the old 6,900 single-run V4 threshold
because this study combines two independent runs per workload. Under the
completed 1B effect and variance scale, 10,000 combined assignments retain a
large margin over the 0.20 materiality boundary; this is a planning statement,
not acquired 3B evidence.

## Acquisition and stop rules

Only after both qualifications pass may a separate authorization freeze and
package four 448-step causal acquisitions. All four must be submitted before
any causal outcome is inspected. Failure closes the relevant authorized stage;
there is no automatic retry, extension, threshold change, workload
substitution, or outcome-guided sample increase.

No EOS launch, model-weight download, training, qualification, or acquisition
is authorized by this design commit.

## Terminal postscript

The later separately authorized four-cell acquisition completed successfully.
OpenMath was `MATERIAL`; GSM8K was `INCONCLUSIVE` around the 0.20 threshold, so
the joint criterion was not met. Both prespecified 3B-minus-1B contrasts were
negative with simultaneous intervals excluding zero, while their cross-workload
difference was inconclusive. These findings support size-related attenuation
for the tested fixed configurations, not a causal parameter-count effect or a
general scaling law. See `terminal_report.md` and `terminal_result.json`.

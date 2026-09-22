# OARS-v2 live-shadow qualification

## Result

The frozen qualification returned
`PASS_SAFE_INSUFFICIENT_NATURAL_CONTENTION`.

The live instrument satisfied every safety and overhead gate:

- 64 learner updates and 64 shadow decisions completed;
- actual selected identities matched eager weight-FIFO at every decision;
- all four scorer proposals satisfied their cardinality and token-band
  contracts;
- skip count and fallback fraction were both zero;
- combined gradient-plus-shadow observer duty was 0.1032%, below the frozen 1%
  maximum; and
- p95 shadow-decision latency was 0.212 ms, 0.00595% of the median learner-step
  interval, below the frozen 1% maximum.

The comparison-support gate did not pass. Every decision had exactly four
ready groups for a four-group batch. Consequently, there were zero contended
decisions, every scorer necessarily proposed the FIFO batch, and the observed
four-of-four overlap contains no evidence that the scheduling signals agree in
a choice-bearing state.

## Interpretation

This is a positive systems result for behavior neutrality and overhead, and a
negative support result for natural eager-FIFO comparison. It is not a result
about training quality or scheduler benefit. OARS-v2 never actuated, and no
training-quality endpoint was analyzed.

Repeating the same qualification is not scientifically justified. The tested
runtime consumes a batch as soon as four groups become ready, so it supplies no
choice set for a four-group alternative policy. The next study must
prospectively declare controlled contention as part of the target environment
and apply an identical admission frontier to FIFO, reward variance, and M4.
That is a new scheduler experiment, not a repair of this one.

## Provenance

The terminal result was opened only after two terminal ZIPs passed integrity,
exact artifact-manifest coverage, declared SHA-256 verification, and
byte-identical-copy checks. The authenticated compact result has SHA-256
`8e10999951aa19a1d548b9cfaa28614ba7efc4a9f647b8dc876c228ced85e23c`.
Large raw artifacts remain outside Git.

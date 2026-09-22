# Repaired controlled-frontier requalification

## Conclusion

The repaired OARS-v2 observer passed the frozen controlled-frontier systems
gate: `PASS_CONTROLLED_FRONTIER_SYSTEMS_READY`.

This establishes that a common eight-ready-group frontier can expose genuine
scheduling choices while preserving controlled FIFO behavior, complete
proposal provenance, and low observer overhead. It does not establish that an
OARS policy improves final training quality because OARS-v2 remained
observe-only and no quality outcome was analyzed.

## Authentication and provenance

- Source commit: `2f017505f90d428dbc968aad7d718826cecb524c`.
- Package commit: `7e54f84a4ae107d1117e597ff3e3b7634e2f0841`.
- Upstream/downstream pipelines: `69327141` / `69327257`.
- Workload/post-run jobs: `451825432` / `451825433`.
- Both terminal ZIPs passed integrity. All six compact payloads had exact
  manifest coverage, matched their declared SHA-256 values, and were
  byte-identical across copies before opening the result.
- Authenticated result SHA-256:
  `cb8db710da1e374e46f5dbb54088cb0fa950cedd0765e9e37d37604e88ace270`.
- Raw archives remain outside Git in the ignored session directory.

## Frozen gate result

- 64 complete learner steps and exactly 64 observer decisions.
- Exactly eight candidates at every decision.
- Every scorer evaluated the exact 70 four-of-eight combinations.
- All 64 acting selections exactly matched controlled FIFO.
- Zero observer skips and zero scorer fallbacks.
- Combined gradient-plus-OARS observer duty: `0.00054854` (0.0549%).
- P95 OARS decision latency: 755,698 ns (0.756 ms), equal to
  `0.00007672` of the median learner-step interval.
- Runtime: 651.56 seconds, within the frozen four-hour limit.

The repair also removed the failure mode seen in the predecessor: the OARS
ledger fell from 181.4 MB and 31,600 poll-time rows to 490 KB and 64 actual
decision rows.

## Choice-bearing evidence

The common frontier produced non-mechanical policy contrasts. Among 64
decisions, age disagreed with FIFO once, reward variance 45 times,
token-normalized M4 46 times, and absolute M4 46 times. Mean four-group overlap
with FIFO was 3.969, 2.859, 2.813, and 2.797 respectively.

These are systems-level proposal statistics, not causal estimates of training
quality. They show that reward variance and M4 define materially different
actions from FIFO under the qualified frontier and therefore can support a
future randomized comparison.

## Next scientific step

Freeze a separate controlled-frontier outcome protocol comparing three acting
policies under the same admission rule:

1. controlled FIFO;
2. reward-variance risk;
3. absolute M4 risk.

Absolute M4 is preferred over carrying both M4 variants because their proposal
behavior was nearly redundant in this qualification. The new study must define
its quality endpoint, seed blocking, pair count or power rule, completion gate,
and failure handling before any outcome launch. This requalification does not
authorize that study.

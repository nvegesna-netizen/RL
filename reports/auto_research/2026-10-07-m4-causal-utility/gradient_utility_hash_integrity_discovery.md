# M4 gradient-utility CountSketch seed-independence discovery

The successful two-group runtime qualification in child pipeline `72207068`
proved that real Megatron gradients can be measured and discarded without any
optimizer, scheduler, learner-version, or parameter change. It also exposed a
pre-acquisition measurement defect in the v1 CountSketch hash.

For the frozen `16,384 = 2^14` bins, the v1 bucket map was

```text
bucket(i, seed) = (A * i + C * seed) mod 2^14
```

where `A` is odd. For any two seeds, this changes every bucket by one constant
rotation while preserving every pairwise collision. The two nominally
independent sketches therefore had identical bucket-collision partitions,
contradicting the frozen design's independent-sketch requirement. The issue is
structural and does not depend on an observed scientific endpoint.

The qualification ledger supplied a second warning. Absolute sketch-norm
relative errors were:

| group | seed 20261019 | seed 20261021 |
|---|---:|---:|
| 1 | 0.129161 | 0.162699 |
| 2 | 0.120372 | 0.122967 |

Both two-group seed medians exceed the frozen full-acquisition median limit of
`0.10`. Two groups are not enough to estimate the full-population median, but
the seed-invariant collision partition is independently decisive: opening 256
groups with v1 would not implement the preregistered instrument.

No 256-group scientific acquisition has started and no construct endpoint has
been inspected. Source commit `b74ce7583` replaces v1 with source-bound
multiply-shift hashing. SHA-256 derives separate odd multipliers and increments
for each `(seed, lane)` pair; high multiply-shift bits select power-of-two
buckets, and a separately derived lane selects Rademacher signs. A unit test
now requires the two frozen seeds to produce different collision partitions
and sign maps while remaining deterministic.

The model, workload, seeds, bin count, group count, hypotheses, endpoints,
bootstrap, decision ladder, and zero-update gates are unchanged. A fresh
two-group qualification must pass both runtime invariants and the frozen sketch
fidelity thresholds before the 256-group acquisition may open.

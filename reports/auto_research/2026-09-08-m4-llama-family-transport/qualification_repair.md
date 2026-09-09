# Llama family-transport qualification repair

Status: `FROZEN_LOCAL_REPAIR_PENDING_NO_TRAINING_PREFLIGHT`.

## Trigger

The first paired qualification closed at one of two green. OpenMath passed all
eight frozen gates. GSM8K completed all 32 steps and passed its capacity,
information, reward-support, and resource gates, but corrected observer duty was
1.2030% against the prospective 1% maximum. Its lifecycle check also required
all opportunity groups to emit `release_delay_completed`; one neutral
zero-second group was instead explicitly removed by bounded shutdown after its
start event.

Neither result is a causal acquisition. Both qualification datasets remain
barred from every estimator, and the original terminal records remain
immutable.

## Repair

The observer measurement scope and 1% ceiling do not change. The synchronous
opportunity calculation retains its existing global float64 arithmetic and
serialized schema. Only sibling reductions are vectorized: valid-token counts
and absolute coefficient sums are reduced across all siblings in two tensor
operations, and scalar extraction is batched. Exact equivalence to the prior
row-wise expressions is a required test.

The neutral lifecycle validator now distinguishes treatment fidelity from
terminal censoring. Every assignment, start, and completion must still encode
the neutral 0.0-second mass-1/1 arm. Opportunity IDs must equal started IDs.
Missing starts or completions are accepted only when the same group has exactly
one later `removed` event whose reason is `bounded_shutdown`; all other missing,
duplicate, non-neutral, or unexplained records fail closed.

## Requalification rule

Because the performance repair changes shared observer source, both OpenMath
and GSM8K must be requalified from the same repaired commit even though the
historical OpenMath qualification passed. First run one shared no-training EOS
preflight. If and only if it passes, freeze and submit both 32-step neutral
qualifications before inspecting either terminal result. Both must independently
pass the unchanged 1% observer-duty and all original capacity/resource gates.

This amendment authorizes no EOS submission, retry, extension, or acquisition.

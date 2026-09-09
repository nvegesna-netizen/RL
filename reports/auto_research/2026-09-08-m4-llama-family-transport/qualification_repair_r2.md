# Llama family-transport final observer repair

Status: `FROZEN_LOCAL_FINAL_REPAIR_PENDING_IMPLEMENTATION`.

## Trigger

The repaired OpenMath and GSM8K neutral workloads both completed 32 trainer
steps, then their shared wrapper failed in post-run summary generation because
the session-local manifest builder called a keyword-only lifecycle API
positionally. Deterministic reconstruction from the preserved artifacts makes
OpenMath green on all intended gates. GSM8K passes every gate except corrected
observer duty: `0.0115385610` against the unchanged `0.01` ceiling.

The first observer optimization reduced GSM8K callback cost from 1.5523 to
1.4583 milliseconds per observation. At the observed cadence, the callback
must be no more than 1.2638 milliseconds merely to reach 1% duty. This final
repair uses a stricter outcome-blind engineering target of 1.0 millisecond,
approximately 0.8% at that cadence.

Neither qualification contains a randomized causal treatment. Its observations
remain barred from all causal estimators, and no Llama causal outcome has been
observed.

## Repair contract

The native GRPO observer may exploit the registered one-prompt, eight-sibling
assignment unit to compute scalar advantages without expanding them to sequence
shape or invoking generic repeated-prompt grouping. It must preserve:

- the production float32 reward, baseline, standard-deviation, and normalized
  scalar-advantage arithmetic;
- the existing float64 coefficient and summary arithmetic;
- every serialized opportunity and lifecycle field;
- the synchronous pre-delay measurement interval;
- identical code for every release arm; and
- the existing assignment, terminal, observer-duty, and causal estimands.

Randomized exact-value equivalence against the pre-repair expanded path is a
mandatory Linux/PyTorch test. Generated qualification manifests must also
exercise the lifecycle assessor through its declared keyword-only API and read
its `supported` result.

## Final stop rule

Run at most one shared no-training EOS preflight after local checks and explicit
user authority. It must pass every equivalence and package test, achieve at
least 25% speedup against the frozen expanded reference, and measure no more
than 1.0 millisecond mean end-to-end observer time on representative 8-by-2048
inputs in the pinned environment. It may not train or qualify.

Only if that preflight is green may exactly one final paired 32-step neutral
qualification be separately frozen and authorized. Both actual qualifications
must pass the unchanged 1% duty ceiling and every original gate. A preflight
miss or either qualification miss permanently closes this Llama branch: no
third repair, threshold relaxation, workload slowdown, single-cell
substitution, retry, or extension is permitted.

This amendment authorizes no qualification or acquisition.

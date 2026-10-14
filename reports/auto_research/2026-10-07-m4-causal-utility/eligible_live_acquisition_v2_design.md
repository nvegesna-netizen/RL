# Fresh eligible-live causal-utility acquisition

Frozen 2026-10-13, before v2 packaging or execution.

## What this experiment answers

Does choosing M4-Shield instead of its reward-variance comparator produce a
better immediate learning update when Shield has an admissible live action?
The experiment moves beyond showing that the scheduler can increase its own
opportunity score: both actions are evaluated on the same withheld data after
one update from the same training state.

The estimand is conditional on the first eligible event within 64 updates,
not the average benefit of enabling Shield throughout training. The endpoint
is held-out clipped-policy-gradient loss, not final answer accuracy. A later
long-horizon trial would answer the latter question.

## Evidence supporting this stage

The original acquisition produced 13 authenticated capsules but did not meet
its 16-of-20 completion requirement. It remains closed; none of its capsules
will enter the new estimate. No original paired outcomes were opened.

A subsequent diagnostic exercised capture at learner version five. The
six-seed qualification then completed 384 learner advances with every recorded
action matching its Shield proposal. The post-run header-count bug was
reproduced and repaired offline without changing training. This establishes
the registered liveness requirements on those six seeds; it does not establish
the root cause or universal absence of the earlier timing-dependent stall.

The separate restore qualification reproduced the same held-out loss in two
fresh processes and preserved parameter hashes. It tested restoration and
non-mutating evaluation, not a complete one-update paired training replay.
That remaining engineering check must be completed before spending compute
on the new cohort.

## Fixed experiment

The machine-readable authority is `eligible_live_acquisition_v2_plan.json`.
It incorporates the unchanged eligibility, capsule, and endpoint contracts
from `eligible_live_decision_causal_protocol.json`; the v1 schema of that
contract is retained for analyzer compatibility, not to reuse v1 assignments.

Use identities `v2a01`--`v2a20`, seeds `20261201`--`20261220`, and the new domain
`m4-eligible-live-causal-acquisition-v2`. Seeds are disjoint from the original
acquisition, restore/capture qualification, and liveness qualification. Each
trajectory contributes at most its first eligible capsule. Trajectories are
independent execution units and can run in parallel.

Eligibility remains an exact eight-group frontier with four-group actions,
distinct Shield/base membership, at least two shared groups, preserved total
and imminent reward-variance sums, the original FIFO-relative token band,
strictly greater imminent opportunity, and no fallback. No observed learning
outcome participates in selection. Capture retains four disjoint,
current-version held-out groups and stops before executing the selected update.

Use the qualified runtime implementation. Turn capture suppression **off**,
retain the 1,200-second progress watchdog, and preserve the four-hour allocated
runtime limit. Set no queue deadline. The watchdog is a progress guard after
execution begins, not a maximum pending-allocation duration.

## Ordered execution gates

1. Validate the frozen design, predecessor hashes, six recovered passes,
   source hashes, and seed separation. Re-run offline recovery on preserved
   raw evidence before recording the execution lock.
2. Build and qualify a one-update replay runner using only an excluded
   historical capsule. Verify policy, optimizer, scheduler, and RNG state;
   equal pre-update losses; exactly one update; and immutable held-out tensors.
   Do not inspect the historical treatment contrast or use it to tune the
   study. Existing loss-only restoration is not a substitute for this test.
3. Package capture and replay sources with new receipts, fresh one-shot guards,
   and clean-room tests. Keep training/scheduler source byte-identical to the
   qualified implementation. Test header-aware classification and immutable
   checksum collection. Verify checkpoint transport without GPU acquisition.
4. Commit the reviewed study package separately from manuscript edits. Source
   the rotated `setup_jet.sh` before EOS access. Submit exactly one
   `runllm.py --no_wait` invocation per capture identity; do not reuse liveness
   guards. Record returned parent/child/workload identities.
5. Monitor status only until all 20 identities are terminal. Authenticate and
   classify every identity. A no-eligible event is not replaced. Apply the
   original minimum of 16 valid capsules; below it, stop without paired replay.
6. If the gate opens, replay both frozen actions once per valid capsule from
   identical state. Keep outcomes sealed until every required arm authenticates.
7. Report every paired contrast, the mean and 95% studentized-bootstrap
   interval, paired-t interval, and exact sign-flip test. Report eligibility
   coverage and failures alongside the conditional estimate; do not assign
   zero learning effects to trajectories without a capsule.

## Artifact transport

The diagnostic checkpoint archive was 13.4 GB, exceeding the downstream
collector's 10 GiB ephemeral-storage budget. Retrieve compact result records
first and checkpoints directly from EOS workload artifacts. Never make the
large archive traverse that collector. Packaging must prove this behavior;
merely retrieving artifacts differently after a failed collector is not a
clean-room transport test.

Inventory workload-owned compact files after they are closed. Inventory capsule
and checkpoint files recursively; retain individual hashes and a binding
manifest. Exclude the still-changing JET wrapper log from the workload checksum
inventory. Large raw artifacts stay outside Git.

## Current stage

Design preparation only. No new capsule, paired outcome, or EOS launch exists.
The next implementation step is the excluded-capsule one-update replay
qualification and clean-room artifact-transport package. These are prerequisites
for launching the fresh cohort, not changes to the scientific endpoint.

## Offline verification

`validate_eligible_live_acquisition_v2_design.py` authenticates five predecessor
records and nine qualified runtime files. Its preserved result is
`eligible_live_acquisition_v2_design_validation.json`, with
`PASS_FROZEN_DESIGN_ONLY` and `launch_ready=false`.

The standard-library test suite passes four tests, including 17 boundary
mutation controls, historical/duplicate seed checks, and evidence-substitution
rejection. Ruff formatting, lint, and whitespace checks pass. Replaying
`recover_eligible_live_v2_liveness_qualification.py` on the retained raw evidence
again reproduces all six repaired passes. These checks require no training
dependencies, network access, or GPU launch.

Reproduce locally:

```bash
uv run --no-project --offline --python python3 reports/auto_research/2026-10-07-m4-causal-utility/validate_eligible_live_acquisition_v2_design.py
uv run --no-project --offline --python python3 reports/auto_research/2026-10-07-m4-causal-utility/test_eligible_live_acquisition_v2_design.py
```

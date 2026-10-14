# Six-seed liveness qualification: offline recovery

Recorded: 2026-10-13

## Finding

All six training processes completed 64 updates and exited with code zero.
Their post-run qualification checker then failed the same assertion. The
checker compared matching scheduling actions with the total number of OARS
ledger rows, which includes one metadata header. Each ledger has 64 decisions
plus that header, so the erroneous check compares 64 with 65.

The original pipelines remain failed and produced no qualification-result
JSON. Their preserved training and decision evidence passes the original
liveness requirements after a one-line offline counting repair. All six
original classifier failures were reproduced on authenticated artifacts, and
all six repaired classifiers pass. No training or scheduler implementation,
seed, assignment domain, update target, timeout, or threshold was changed.

## Per-identity evidence

| Identity | Seed | EOS job | Learner advances | Matching actions | Suppressed eligible decisions |
| --- | --- | --- | --- | --- | --- |
| lq01 | 20261102 | 482163040 | 64 | 64/64 | 2 |
| lq02 | 20261103 | 482168582 | 64 | 64/64 | 5 |
| lq03 | 20261109 | 482170445 | 64 | 64/64 | 6 |
| lq04 | 20261110 | 482172905 | 64 | 64/64 | 4 |
| lq05 | 20261111 | 482174214 | 64 | 64/64 | 10 |
| lq06 | 20261119 | 482175440 | 64 | 64/64 | 5 |

Each run has learner versions exactly 1 through 64, a controller summary with
`train_steps=64`, `trainer_version=64`, and
`PASS_FULL_64_UPDATE_LIVENESS`, and 64 actual action lists independently
matching their M4-Shield proposals. Suppression counts agree with eligible
proposal differences. All 25 focused tests passed in every run: 2 capsule,
7 OARS, and 16 controller tests.

The original classifier reached its action-count assertion only after
checking run exit code zero, absence of the timeout record and forbidden
capsule, nonempty audit ledgers, controller qualification, suppression counts,
and the complete sequence of learner advances. Independent offline replay
rechecks those requirements. Controller records also report no checkpoint or
post-update held-out outcome written/opened.

These failures occurred after approximately 10--12 minutes of allocated
execution. Slurm records `FAILED` with exit code `1:0`, rather than `TIMEOUT`.
The roughly 43--45-hour queue waits ended in allocation. This was neither a
queue deadline nor the four-hour workload limit.

## Authentication and recovery

The lifecycle, opportunity, OARS, observer-duty, training log, and focused-test
log from each EOS job match their workload-owned SHA-256 declarations. Those
six files and the checksum ledger are byte-identical in the corresponding
logs-after artifacts. All six submitted manifest hashes match the pre-launch
package receipt; recovery extracts the actual classifier from those manifests.

The recursive checksum sweep also captured
`jet_assets/output_logs/output_script-0.log`, the JET wrapper output stream.
That file continued changing after its checksum was recorded. Its final bytes
are identical across the preserved copies, but differ from its earlier hash.
This mismatch is recorded explicitly and is not waived for any workload-owned
artifact. Future collection now hashes top-level workload files, excluding
the mutable harness subdirectory.

The separate recovery result is
`eligible_live_v2_liveness_qualification_offline_recovery.json`, with status
`PASS_ALL_SIX_ORIGINAL_LIVENESS_REQUIREMENTS_OFFLINE_RECOVERED`. It preserves
the original failed pipeline state, absent original result files, source and
protocol identity, manifest hashes, every evidence-file hash, and each
original/repaired classifier replay. It does not overwrite original artifacts.

Replay command:

```text
python reports/auto_research/2026-10-07-m4-causal-utility/recover_eligible_live_v2_liveness_qualification.py
```

The original GPU workloads need no relaunch. The liveness evidence is an
engineering prerequisite; no causal utility or terminal-accuracy estimate is
produced by this qualification. Paired replay and the fresh v2 acquisition
have not started. The next scientific step is to freeze the prospective v2
capture design with fresh seeds, then use its authenticated completion gate
before opening paired outcomes.

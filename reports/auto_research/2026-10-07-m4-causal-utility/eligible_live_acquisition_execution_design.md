# Eligible-live paired acquisition execution design

The causal acquisition begins with 20 independent, outcome-excluded capture
trajectories (`a01` through `a20`, seeds `20261101` through `20261120`). Each
trajectory may contribute at most its first eligible M4-Shield decision and
stops after 64 completed updates if none occurs. The seed list was fixed before
the first acquisition launch; qualification seed `20261031` is not reused.

The capture jobs are scientifically independent and may run in parallel. Each
capsule must bind its trajectory identity, GRPO seed, and common assignment
domain in its authenticated manifest. At the captured decision, the selected
update is not executed and no base-versus-Shield post-update outcome is
computed. A trajectory with no eligible event is a valid no-capsule result and
is not replaced.

Only after all 20 capture identities are terminal and authenticated is the
completion gate evaluated. At least 16 valid capsules are required. If the gate
fails, the study stops without paired replay. If it passes, every valid capsule
is replayed from its identical pre-update checkpoint for exactly one base update
and one Shield update. No rollout regeneration or action reselection is allowed.

Replay results remain unopened until every valid capsule has two authenticated
arm records. The registered analysis then reports every paired contrast and the
paired mean with its 95% studentized-bootstrap interval, paired-t interval, and
exact sign-flip test. Pipeline status is execution provenance, not a scientific
endpoint.

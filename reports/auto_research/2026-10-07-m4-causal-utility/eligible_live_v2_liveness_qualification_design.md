# Eligible-live v2 multi-seed liveness qualification

The single-seed diagnostic completed but captured an eligible capsule at update
five. It therefore validated the watchdog and capture path without exercising
the full 64-update no-capsule lifecycle that failed on six v1 trajectories.
That result is insufficient to infer that the timing-dependent failure is
absent.

This qualification targets exactly the six v1 seeds that allocated and then
timed out. Seed `20261105` is excluded because its predecessor never allocated.
The qualification keeps controlled release, OARS-v2 scoring, M4-Shield
actuation, rollout generation, optimizer updates, scheduler updates, and
weight synchronization unchanged. A default-off qualification flag consumes
and records any eligible capsule trigger but suppresses capture termination.
It does not change the selected groups. Consequently, each trajectory must
exercise the ordinary controller through all 64 updates.

The flag is an engineering qualification hook, not a proposed production
policy. It is valid only with the eligible-live watchdog, requires the frozen
64-update configuration, and forbids capsule and checkpoint output. Every run
must finish at learner step and version 64 with complete compact audit ledgers.
A watchdog timeout or any other error fails that identity closed.

The six independent runs may be submitted in parallel. Each has one attempt,
a four-hour allocated-runtime limit, and no client-side queue deadline. All six
must pass and authenticate before a fresh v2 causal acquisition can be frozen.
The qualification opens no held-out post-update outcome and estimates no
causal or terminal-quality effect.

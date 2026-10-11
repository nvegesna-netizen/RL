# Eligible-live acquisition v1 terminal report

Acquisition v1 is terminal and failed its prospective completion gate. Thirteen
of the 20 frozen trajectories produced authenticated, outcome-excluded capsules;
the protocol required at least 16 before paired replay. Six allocated workloads
timed out and one workload never allocated. No paired replay was opened and no
post-update arm outcome was inspected.

Every completed result was retrieved directly from its EOS job artifact and
checked against that job's SHA-256 manifest. All 13 results bind the expected
trajectory identity, seed, and `m4-eligible-live-causal-acquisition-v1` domain;
all embedded capture checks pass; and all report that post-update outcomes
remain unopened. The compact hashes and execution identities are preserved in
`eligible_live_acquisition_v1_terminal_authentication.json`.

## Why six allocated workloads timed out

The allocated failures were not slow training runs. Their completed updates
took approximately 4--16 seconds each. They then stopped emitting controller
progress after updates 2, 6, 6, 2, 2, and 10 respectively, remained allocated,
and were terminated by Slurm at approximately four hours. Their traces contain
no Python exception, out-of-memory event, NCCL failure, or failed runtime
authority check. In contrast, successful trajectory `a01` completed 11 updates
and captured its eligible decision in about eight minutes of allocated runtime.

The evidence localizes the failure to a timing-dependent liveness stall between
the rollout and training pumps. The controlled frontier requires eight ready
groups, while rollout admission, stale-group replenishment, and selection are
separately gated. The terminal traces are consistent with training waiting for
a complete selectable frontier while rollout progress has stopped. Because
Slurm terminated the processes before graceful cleanup, the lifecycle recorders
did not flush and no task stack was captured; therefore the exact blocked await
must be reproduced under watchdog instrumentation before a code repair is
accepted.

Trajectory `a05` has a separate operational cause. An unintended completion
deadline expired while it was pending, so it received no allocation and ran for
zero seconds.

## Scientific disposition

The original frozen rules classify integrity or runtime failure as fail-closed,
forbid automatic retry or replacement, and stop without replay below 16
authenticated capsules. Those rules are applied without amendment. Acquisition
v1 therefore makes no causal-utility claim.

Any further acquisition must be a separately frozen v2 study. It must first
reproduce and repair the liveness failure, demonstrate a forced no-eligibility
path through all 64 updates within EOS's four-hour runtime limit, and rerun a
complete cohort under one implementation. The 13 v1 capsules may be retained as
supporting engineering evidence but are not pooled into the v2 primary analysis.

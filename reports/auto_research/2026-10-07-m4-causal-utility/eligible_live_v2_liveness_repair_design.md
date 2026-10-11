# Eligible-live v2 liveness-repair design

Acquisition v1 is terminal, and its 13 successful capsules are not eligible for
the v2 primary analysis. Six allocated trajectories stopped after 2--10 learner
updates and reached EOS's four-hour runtime limit without a Python, OOM, NCCL,
or integrity error. Because Slurm killed the processes, graceful cleanup did
not preserve controller stacks or lifecycle ledgers. This evidence supports a
timing-dependent controller-liveness hypothesis, but not a particular blocked
await.

The first v2 action is therefore diagnosis, not selective scientific retry.
The default-off watchdog is enabled only for eligible-live capture. It observes
learner-step/version changes and explicit capture-phase transitions. After
1,200 seconds without any such progress, it atomically writes bounded
controller, replay-buffer, scheduler, and asyncio-stack state, then raises
through the existing cleanup path so research ledgers can flush. It records no
candidate content, held-out value, post-update value, arm outcome, or causal
contrast.

One diagnostic run uses seed `20261102`, whose v1 predecessor stopped after
update 2. A captured timeout, an outcome-excluded capsule, or completion of all
64 updates without eligibility is a valid diagnostic terminal state. None is a
scientific endpoint. A failure without one of those records fails the
instrument.

After a captured timeout, the exact blocked state must be converted into a
minimal repair and a deterministic regression. The repaired implementation
must then pass a prospectively frozen, multi-seed forced-no-eligibility
qualification through all 64 updates. Only after that qualification may a new
20-trajectory v2 acquisition be frozen with new seeds. V1 records remain
historical engineering evidence and are not pooled into the v2 estimate.

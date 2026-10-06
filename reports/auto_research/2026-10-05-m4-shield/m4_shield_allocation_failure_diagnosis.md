# M4-Shield allocation-failure diagnosis

## Classification

`AUTHENTICATED_PRE_EXECUTION_ALLOCATION_FAILURE`

The submitted M4-Shield qualification did not execute. Slurm job `6171790`
remained pending for priority for 4:00:21 and terminated in state `DEADLINE`.
Its recorded workload duration was `00:00:00`, and Slurm had no step data.

This is not an M4-Shield implementation failure, preflight failure, training
failure, or qualification-gate result.

## Evidence

- Upstream pipeline `71774486`: failed only because its downstream bridge
  propagated failure; generator job `470864332` succeeded.
- Downstream pipeline `71775100`: workload `470868940` failed; both log jobs
  succeeded.
- The terminal workload archive contains only JET and Slurm log metadata. Its
  `slurm.out` is empty and it contains no `workspace/assets` files.
- The logs-after archive contains the terminal JET event record, but no runtime
  output, preflight result, training log, OARS ledger, or qualification result.
- Both downloaded archives are traversal-safe. Their common pre-execution
  records authenticate byte-for-byte.

The authenticated JET event says:

- Slurm state: `DEADLINE`
- Queue time: `04:00:21`
- Workload duration: `00:00:00`
- Exit code: `1:0`
- Slurm step data: absent

## Scheduler boundary

The submitted manifest correctly set only `spec.time_limit = 14400`, producing
the intended four-hour workload limit. It supplied no queue-deadline override.
JET added its normal `--deadline now+8hours` flag, but EOS terminated this
pending allocation after approximately four hours before that deadline. No
scientific or runtime parameter caused the failure.

## Scientific disposition

No scientific attempt was consumed: zero learner steps and zero scheduler
decisions occurred, and the frozen qualification gate never ran. The launcher
attempt was consumed, however, and the frozen protocol forbids an automatic
retry.

The defensible continuation is one separately authorized replacement
submission of the byte-identical manifest
`a14ca17f4547eba2bbff6a4c87cb3bbd4484f2aed71b6be77678073eeb778f97`.
No source, policy, protocol, workload time limit, or queue-deadline override
should change. Queue conditions—not code—must provide the new opportunity to
allocate.

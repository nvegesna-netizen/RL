# Repaired enacted-OARS qualification failure

The repaired enacted-OARS qualification is closed as a **systems liveness
failure with no scientific result**. The dependency repair worked: the
TransferQueue controller initialized, OARS reached execution, and training
completed two of 64 updates. The run then stopped making progress and Slurm
terminated it at the frozen four-hour limit.

This was not a queue-allocation failure. Slurm allocated the job after 53
seconds. The workload ran for 3:59:44, the whole job ended in `TIMEOUT`, and
the workload step ended `CANCELLED` with exit `0:15`.

Both terminal ZIP archives passed integrity checks. Their copies of the only
declared run log were byte-identical. The log was opened only after that
cross-copy authentication. No result JSON, terminal SHA-256 manifest, or
scientific ledger was preserved, so no qualification or training-quality
endpoint exists.

## Failure mechanism

The authenticated run log records training steps 1 and 2, then ends with:

> `evicted 1 stale prompt group(s)`

There is no Python traceback or later application progress. The code and
frozen configuration provide a high-confidence liveness diagnosis:

1. Each training step consumes four prompt groups.
2. The weight-FIFO baseline permits one version of lookahead, giving exactly
   eight buffer slots, and the qualification requires eight ready candidates
   before selection.
3. Enacted OARS may choose four groups across the two weight versions rather
   than draining the oldest four as FIFO would.
4. If one unselected old group becomes stale at the next trainer version,
   eviction leaves three survivors. One newly admitted four-group batch raises
   the ready count to seven.
5. Seven is below the eight-candidate watermark. A second whole batch cannot
   be admitted until the trainer advances, while the trainer cannot advance
   until selection sees eight candidates.

That circular wait exactly matches the final eviction message followed by
nearly four hours of silence. The terminal buffer ledger is unavailable, so
the receipt labels this as a high-confidence code-path reconstruction rather
than claiming a directly captured buffer snapshot.

Increasing the time limit would not repair this state. The necessary next
engineering step is a CPU-only, multi-cycle liveness reproducer covering
cross-version OARS selection, stale eviction, the eight-candidate watermark,
and whole-batch admission. A repair must preserve the prospectively fixed
eight-candidate opportunity set; simply enlarging the buffer or candidate set
would change the intervention.

## Protocol disposition

The one authorized replacement attempt is consumed. No automatic retry or
extension is authorized, the confirmatory acquisition remains locked, and the
two partial updates must not be interpreted scientifically. Any repair and
new qualification require a separately versioned plan and explicit authority.

Canonical machine-readable evidence is in
`qualification_act_replacement_terminal_authentication.json` and
`qualification_act_replacement_failure_receipt.json`.

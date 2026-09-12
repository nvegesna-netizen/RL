# Allocation-recovery terminal diagnosis

All 14 recovery workloads are terminal operational failures, not scientific
results. Every workload first waited in `PENDING (Priority)`, obtained an EOS
node, ran for approximately one hour, and was then cancelled by Slurm with
signal 15 and `DUE TO TIME LIMIT`. No scientific output artifact was opened,
and no relaunch was performed during this diagnosis.

The repair removed both explicit runtime fields. That did remove the prior
four-hour request, but it did not remove or lengthen the queue deadline. JET
rendered each recovery submission with:

```text
--time 0:3600 --deadline now+8hours
```

For comparison, the successful original `b01_immediate` workload rendered a
four-hour runtime request, retained the same eight-hour queue deadline, and
completed normally in `01:36:23`:

```text
--time 0:14400 ... --deadline now+8hours --time 04:00:00
```

The representative recovery workload `b10_immediate` queued for `01:40:22`,
ran for `01:00:12`, and ended `TIMEOUT`; its workload step ran `00:59:53` and
ended `CANCELLED` with exit `0:15`. The other 13 workload steps have the same
signature, with durations from `00:59:44` to `00:59:55`.

## Conclusion and prospective repair

The cause is the one-hour default runtime that JET inserted after all explicit
runtime fields were removed. This is distinct from the original four-hour
allocation-wait deadline failure: these 14 jobs did allocate and execute.

A future separately authorized recovery should restore only the manifest-level
runtime envelope, `spec.time_limit = 14400`, and leave custom `sbatch` and
`srun` time flags unset. That should render exactly one `--time 0:14400` flag
while preserving the existing `--deadline now+8hours`. A clean-room validation
must assert those rendered scheduler arguments before submission. The
scientific scripts and identities must otherwise remain unchanged.

The recovery attempt remains operationally censored, the outcome embargo stays
in force, and the completion gate remains locked.

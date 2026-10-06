# M4-Shield replacement deadline-failure diagnosis

## Classification

`AUTHENTICATED_PRE_EXECUTION_ALLOCATION_DEADLINE_FAILURE`

The allocation-repaired qualification did not execute. Slurm job `6174426`
remained pending for priority for 6:00:18 and terminated in state `DEADLINE`.
Its workload duration was `00:00:00`.

This is not an M4-Shield implementation, preflight, training, or qualification
failure.

## What the first repair accomplished

The actual `sbatch` command contained no `--time` option. This authenticates
that `spec.time_limit: null` successfully removed the explicit four-hour time
limit. EOS assigned its default `TimeLimit=02:00:00`.

JET separately injected `--deadline now+8hours`. The job expired after almost
exactly six queued hours: the eight-hour completion deadline minus the default
two-hour runtime limit. This timing strongly identifies the remaining deadline
as the allocation cutoff.

## Execution boundary

The workload and logs-after artifacts authenticate the same job and event
records. `slurm.out` is empty. There is no runtime source extraction, preflight
output, training log, OARS ledger, observer-duty record, scheduler decision, or
qualification result. Therefore this consumed the replacement launcher
attempt but no scientific attempt.

## Defensible repair

Keep the validated manifest and every frozen scientific/runtime payload
unchanged. Change only the launcher custom configuration from `{}` to the
already established repository configuration:

```json
{
  "launchers": {
    "dgxh100_eos": {
      "sbatch_additional_flags": {
        "deadline": false
      }
    }
  }
}
```

Before any separately authorized submission, rebuild the authorization and
package receipt so they cite this terminal failure, validate the generated
configuration, and require both no `--time` and no `--deadline` in the actual
Slurm command. No source, seed, workload, controller, analyzer, or scientific
threshold should change.

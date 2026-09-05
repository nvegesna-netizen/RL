# GSM8K neutral qualification R6 result

Status: `FAILED_SLURM_DEADLINE_BEFORE_ALLOCATION`; the repaired workload never
started and no retry occurred.

The single authorized R6 release submitted manifest SHA-256
`514c369a0fa68dd110cff29806760092362df21467d5e46ecf3d3d78515ffc83`
through `runllm.py --no_wait`. Parent pipeline `66330380` generated downstream
pipeline `66330524`. Generator `426666664`, logs-before `426667739`, and
logs-after `426667741` succeeded. EOS compute `426667740` failed, making both
pipelines terminal `failed`.

EOS submitted Slurm job `5978609` to `batch` with `--no-requeue`, a four-hour
runtime request, and JET's fixed deadline of eight hours after submission. The
job remained pending for `Priority`. After `04:00:19` of queue time, Slurm
reported:

```text
state: DEADLINE
runtime: 00:00:00
exit code: 1:0
```

The timing explains the cancellation: once four hours remained before the
eight-hour deadline, Slurm could no longer guarantee completion of a requested
four-hour allocation. It canceled the job without allocating a node. The
artifact therefore has no workload assets and an empty `slurm.out`; the R6
fingerprint/Megatron bootstrap gate, config validation, model initialization,
and training were never reached.

This result is not evidence against the R6 bootstrap repair, GSM8K
compatibility, the neutral instrument, or the M4 hypothesis. It is solely a
scheduler-feasibility failure. A proper successor should retain the four-hour
confirmatory scientific ceiling but request a shorter qualification allocation,
leaving more of JET's fixed deadline available for priority queueing. A
90-minute qualification request is proportionate: R5 reached its first training
result in approximately 44 minutes after a roughly 33-minute Slurm queue, while
the accepted qualification's measured workload duration was 984.588 seconds.

The preserved 18,926,485-byte artifact remains outside Git at
`session/20260903_m4_qwen3_1p7b_transport/gsm8k-qualification-r6-66330524-job-426667740.zip`
with SHA-256
`28b5e3668850efda3b9fc26477e319cccdf2c180d4744d4c75ecad4de16bd5f7`.

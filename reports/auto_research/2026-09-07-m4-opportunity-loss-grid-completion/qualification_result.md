# Qwen3-0.6B/GSM8K neutral qualification result

Status: `QUALIFICATION_GREEN`.

The single authorized 32-step neutral qualification succeeded. Parent pipeline
`66666794`, downstream pipeline `66667017`, generator job `429466569`,
logs-before job `429468110`, compute job `429468112`, and logs-after job
`429468114` all completed successfully. Slurm job `5989573`, the workload, and
rank 0 exited zero.

Independent reconciliation of the raw terminal streams confirmed all registered
qualification gates. The run completed trainer versions 1–32 and produced 565
unique opportunity groups over start versions 0–31, or 17.65625 groups per
observed version against the minimum of 15. Rewards included both 0 and 1. All
565 opportunity groups received and completed the sole neutral, zero-second
release arm. The lifecycle contains 581 assignments in total; the additional
assignments were outside the terminal opportunity-group set and do not alter the
registered information-yield calculation.

Corrected observer duty was `0.002934918347246835`, below the `0.01` ceiling.
Using the preregistered 1.25 safety factor, the measured step cadence and fixed
overhead project the 558-step acquisition to 1.538276939 wall-hours and
3.076553878 two-GPU-hours, below the four-wall-hour and eight-GPU-hour caps.

The terminal ZIP has 38 members, passes ZIP integrity, is 106,453,609 bytes,
and has SHA-256
`bc3931a37c8e9b510392806d552e4043c2af55bc3ed91fd3f864f4f768213677`.
Its embedded five-file checksum manifest also reconciles exactly.

This is a resource and instrument qualification, not a scientific result. It
used a single neutral arm, produced no causal estimate, and its observations are
excluded from the primary and grid estimators. The next gate is to freeze and
review the 558-step acquisition package. A separate exact authorization remains
required before one `runllm.py --no_wait` acquisition submission.

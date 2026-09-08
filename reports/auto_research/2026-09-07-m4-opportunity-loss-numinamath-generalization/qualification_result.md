# NuminaMath paired neutral qualification result

Status: `PAIRED_QUALIFICATION_GREEN`.

Both prospectively matched neutral 32-step qualifications succeeded. The
Qwen3-0.6B parent/downstream pipelines are `66811941`/`66812105`, and the
Qwen3-1.7B parent/downstream pipelines are `66811965`/`66812188`. Slurm jobs
`5994668` and `5994670` completed with zero exit status.

The 0.6B cell completed 32 steps and yielded 552 opportunity groups over 32
observed start versions. Its corrected observer duty was
`0.0023387151297210233`. With the frozen 1.25 safety factor, its 448-step
projection is `1.8656132385566304` wall-hours and `3.731226477113261`
two-GPU-hours.

The 1.7B cell completed 32 steps and yielded 535 opportunity groups over 31
observed start versions. Its corrected observer duty was
`0.0017940087188015518`. Its 448-step projection is
`2.0948084911506943` wall-hours and `4.189616982301389` two-GPU-hours.

Both cells observed rewards `{0, 1}`, exceeded 15 opportunity groups per
observed start version, used only the neutral zero-second arm, kept corrected
observer duty below 1%, and project below the four-wall-hour and eight-GPU-hour
per-cell ceilings. All embedded hashes and both 38-member ZIPs pass independent
reconciliation. Artifact SHA-256 values are
`9df74eaa4afb49b89ef069527d9f2fece5c94c410c0078b2fc3f732d0c9df5e6`
for 0.6B and
`3e179fc030d8b2ced181b62817fa426b3b81f4ee02e599e195edef9db36549a7`
for 1.7B.

This is an operational qualification, not a causal result. Qualification
observations cannot enter any causal estimator. No retry, extension, causal
estimate, or scientific acquisition occurred. The next gate is to freeze both
448-step acquisition packages before submitting either one; acquisition needs
fresh explicit authorization.

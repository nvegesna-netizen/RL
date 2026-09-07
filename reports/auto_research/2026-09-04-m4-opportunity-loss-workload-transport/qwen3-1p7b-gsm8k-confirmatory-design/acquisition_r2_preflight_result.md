# Qwen3-1.7B GSM8K R2 no-training preflight result

Status: `SUCCESS_R2_NO_TRAINING_PREFLIGHT_GREEN`.

Parent pipeline `66562916` and downstream pipeline `66563008` succeeded. The
intended EOS compute job `428610191` succeeded under Slurm job `5986527` with
exit code 0. Logs-before job `428610190` and logs-after job `428610192` also
succeeded. Exactly one submission was made and automatic retry and extension
remained disabled.

The terminal artifact is
`r2-preflight-66563008-job-428610192.zip`, SHA-256
`6c2cb9ec94cf3ab944d6c8a6eb6c3686761c1d55c4580d9e99e7116c31ea69e4`,
size 104,246,593 bytes. Its ZIP integrity check passed. The preserved summary,
SHA-256
`68ce03122491cdf6590adf585bc81be449813a0e95014b3eb00c63290c8e891f`,
reports `preflight_green=true`, exact config resolution, 136 selected tests
passing with 17 warnings, and no training or acquisition start.

The emitted lock has SHA-256
`db51df4da5309abeb4622604499aeed5f47066c0060bc491915b2ceef092b954`.
All 37 file hashes and sizes in that lock were independently reconciled against
source commit `b79e419aea95b5226ac4274c9a1535251658b5a5`. The lock binds the two-epoch
R2 config, fresh assignment identity, epoch-specific assignment unit, exclusion
of R1 observations, 558-step target, minimum 7,395 primary assignments, frozen
protocol, R1 failure record, accepted M4 evidence, and pinned image.

The runtime emitted a code-version warning because the dependency image and
the source archive have different commits. This is expected in this package:
the image commit, image hash, source commit, and source-archive hash are all
separately pinned, the archive hash was checked before extraction, and the
tests and lock ran against the extracted source through `PYTHONPATH`.

This result closes only the R2 no-training preflight gate. It verifies that the
capacity repair is represented and accepted by the pinned environment; it does
not empirically show that a two-epoch acquisition reaches 558 steps. The next
step is to build and independently inspect a one-shot acquisition package bound
to this terminal artifact. That later release requires acquisition authority
and must retain the four-hour, two-GPU cap with no retry or extension.

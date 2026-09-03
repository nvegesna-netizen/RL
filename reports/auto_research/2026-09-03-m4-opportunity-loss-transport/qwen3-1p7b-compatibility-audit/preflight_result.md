# Qwen3-1.7B neutral qualification preflight result

Status: `PREFLIGHT_GREEN`; qualification and scientific acquisition have not
started.

The pinned, no-training EOS preflight completed successfully on 2026-09-03.
Parent pipeline `66114094`, downstream pipeline `66114826`, generator job
`424788512`, and EOS compute job `424793998` all reached terminal success. The
submitted workload identity was exactly
`basic/m4-qwen3-1p7b-neutral-no-training-preflight dgxh100_eos 00 [2 dgxh100_eos]`.

The artifact reports exact configuration resolution, 120 selected tests passing,
Ruff format and checks passing, and Python compilation passing. Its fail-closed
lock SHA-256 is
`b45bf7a041e1b9c7a3bf08032c4c4959b34c8dab9062965a77603c3a1e0a4fb1`.
The preserved terminal artifact is 47,046,815 bytes with SHA-256
`e3cccb5ad47efcc028ca52334d155b2919c254a877c019c45721cc7653ea8c3f`.

The artifact explicitly records `training_started=false`,
`qualification_started=false`, `acquisition_started=false`, and
`automatic_retry=false`. Pipeline success therefore means that the compatibility
gate passed; it is not a runtime qualification or a scientific result.

The log emitted a container/code fingerprint warning because the source archive
does not carry Git submodule worktrees. The pinned container supplied those
dependency revisions, and exact config resolution plus all selected tests passed.
This is non-blocking for the no-training gate, but the gate does not establish
model-memory fit, throughput, observer duty, or terminal training completeness.

The next scientific step is one separately authorized, frozen 32-step neutral
qualification. Its outputs remain limited to resource fit, throughput, observer
duty, and completeness. It cannot estimate the control-versus-d5 effect and its
data cannot enter a later confirmatory estimator.

# GSM8K neutral qualification submission result

Status: `FAILED_JET_TEMPLATE_BRACE_ESCAPING`; no qualification ran.

The exactly-once release submitted manifest SHA-256
`2d3d2a889bddba32723058c8b89ddc1b0e8c530c5ce974e2c5abe367a36d3185`
with `runllm.py --no_wait`. Parent pipeline `66257178` failed in generator job
`426025859`; bridge job `426025860` was skipped. No downstream pipeline, EOS
compute job, Slurm allocation, qualification training, or scientific
acquisition was created.

The failure was in packaging, not the study configuration. The script retained
literal Python, shell, and JSON braces while reserving only `{assets_dir}` for
JET substitution. JET normalized the workload but failed before emitting the
downstream pipeline definition. The preserved generator artifact is 9,398,265
bytes with SHA-256
`934ceaa73d0259effdea2a8c6343724371c42e775a091230adafbad16af25dc1`.

The builder now escapes every literal brace before restoring the single JET
placeholder. The corrected replacement manifest SHA-256 is
`cb0a23720d218251662fd4a22b054a6e298eec1d8f538c999631a91cf25940b3`.
Its local validation includes an actual formatting pass followed by Bash and
embedded-Python syntax checks. The first guard remains consumed; the corrected
manifest will not be submitted without a new explicit release.

# Qwen3-1.7B Confirmatory Replacement Preflight Result

The authorized replacement no-training preflight reached the intended two-GPU
EOS job and confirmed the validator correction: all 135 selected tests passed,
including the confirmatory lock tests. It then stopped at `ruff format --check`
because the corrected expression had not been normalized by Ruff's formatter.

This is a source-format gate failure, not a scientific-design, model-topology,
or runtime-compatibility failure. Pipeline `66169109`, job `425284574`, ended
with `script_failure`. Its 47,114,959-byte artifact has SHA-256
`3fefca25624e77931fab4364ea30432e6f0ad0cc3a9d78fdb13a6edd5da5215c`.
No training or scientific acquisition began, and no automatic retry occurred.

The local source has now been formatter-normalized. A new commit, source
archive, manifest, and explicit authority would be required before another
preflight. The frozen scientific protocol remains unchanged.

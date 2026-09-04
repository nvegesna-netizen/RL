# Qwen3-1.7B GSM8K replacement workload preflight result

Status: `TERMINAL_PREFLIGHT_GREEN`.

The separately authorized, exactly-once replacement no-training preflight
reached the intended EOS workload. Parent pipeline `66251881` generated
downstream pipeline `66251989`; EOS compute job `425987171` ran as Slurm job
`5975864` with the exact workload identity
`basic/m4-qwen3-1p7b-gsm8k-workload-no-training-preflight dgxh100_eos 00 [2 dgxh100_eos]`.

The safe archive gate passed. All 123 selected tests passed, including the
corrected analyzer-isolation test. Configuration resolution, Ruff format and
lint checks, Python compilation, canonical lock generation, and the terminal
`M4_QWEN3_1P7B_GSM8K_WORKLOAD_PREFLIGHT_GREEN` marker all passed. The rank,
workload, and Slurm exit codes were zero.

The preserved compute artifact is 47,176,462 bytes with SHA-256
`94c21c7d8fea68e7c698df94d52c153ca6c7ce324edf0c87fa951c30d63d389d`.
Its canonical lock SHA-256 is
`5ced4c617c5f0154a2b03b89262583e94b884b6de2f683470cbac984eff99991`.
The lock binds source commit `933802498b3e5e0ce392bff7b448854a46f3ef44`,
source archive SHA-256
`41d08ca7ebdd5fb7eaf1a62255d5ee4cd5e5091f12f5eaa0b3b928bd57201b11`,
protocol SHA-256
`84c43e4fc32a7f647505729a556d5e0fb088a38fd7eedd1ea235dd51a4ec3fc7`,
and dataset-overlay SHA-256
`b6ed0135388ceb7bb272fc6263f3d97955779949f22cd394696c05d764b9d3c7`.

This result closes only the no-training preflight gate. Training,
qualification, and scientific acquisition did not start, and no causal result
was produced. Automatic retry and extension remained disabled. The next gate
is a separately frozen and separately authorized 32-step neutral GSM8K
qualification; it must remain operational evidence and may not enter or resize
the fixed confirmatory estimator.

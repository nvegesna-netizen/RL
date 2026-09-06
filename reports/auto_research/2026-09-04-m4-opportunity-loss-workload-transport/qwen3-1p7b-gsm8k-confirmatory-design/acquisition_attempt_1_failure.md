# GSM8K confirmatory acquisition attempt 1: terminal failure

Status: `FAILED_BEFORE_ACQUISITION_NO_SCIENTIFIC_RESULT`.

## Terminal evidence

The exactly-once submission created parent pipeline `66461092`, downstream
pipeline `66461158`, and EOS compute job `427764750`. The generator and both log
jobs succeeded, but the compute job and both pipelines terminated failed. Slurm
job `5983488` ran on `eos0514` with rank exit code 1. The preserved compute
artifact is 47,241,819 bytes with SHA-256
`971c8479e55e5d4031ce6d09fe8c9dbff41f817d71c4ee76f6509c31c2716f01`.

The artifact is a valid ZIP. Its 12,511,706-byte output log has SHA-256
`ab37b7617a19407d8ae0729af3352d942c58e7e2a5380e624a2c8942aac89857`;
the Slurm log has SHA-256
`f6d70805a5432e15caa890da5ebd884d3424250f1107a00a9bed1a641a55cea3`.
The safe archive gate passed with 1,910 unique members and 16 validated
symlinks. All embedded source, contract, delta, qualification, authorization,
preflight-lock, and preflight-summary hashes matched.

## Failure mechanism

The next gate attempted `import megatron` and raised
`ModuleNotFoundError: No module named 'megatron'`. The terminal-green R7 builder
performed `import nemo_rl` immediately before `import megatron`; the acquisition
builder omitted the first import. Importing `nemo_rl` registers the vendored
Megatron path for the isolated run tree. The container baseline and frozen
source overlay were present, and the Megatron sentinel file existed, so this is
an import-order error in the package gate rather than missing frozen source or
an unavailable submodule.

## Scientific interpretation

The failure preceded the fingerprint/Megatron pass, frozen-authority check,
configuration check, acquisition-start marker, and training entrypoint. The
artifact contains no acquisition run log, lifecycle ledger, opportunity ledger,
observer-duty record, analysis result, acquisition summary, or scientific
checksum ledger. Therefore zero confirmatory trainer steps completed, no
control:d5 observations entered the registered estimator, and no causal estimate
exists.

This attempt neither supports nor refutes a material M4 opportunity-loss effect
on GSM8K. R7 remains a successful resource qualification, but qualification
data remain excluded and cannot answer the causal question. The GSM8K study is
not scientifically closed.

## Boundary for any successor

Attempt 1 authority is consumed. A successor would need to reproduce the exact
R7 bootstrap import order, validate that narrow package-only repair, freeze a
new manifest and guard, and receive fresh explicit authorization. The protocol,
source archive, model, GSM8K workload, control:d5 arms, 558-step geometry,
estimator, thresholds, and no-retry/no-extension rules should remain unchanged.

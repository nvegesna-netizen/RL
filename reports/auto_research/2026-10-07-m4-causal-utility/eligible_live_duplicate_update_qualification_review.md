# Duplicate comparator-update qualification review

2026-10-14. Engineering qualification only; the historical capsule is excluded
from every new primary estimate.

## Evidenced repair

`nemo_rl/models/megatron/setup.py` explicitly constructs checkpoint settings
with `load_rng=False`. The old restore test requested optimizer loading and
verified model weights and forward-only loss; it did not exercise a training
step or verify restored moments/master weights/RNG. The frozen historical
checkpoint's authenticated `run_config.yaml` confirms `save_rng=true`,
`save_optim=true`, and distributed optimizer use.

Use the existing `worker_extension_cls_fqn` interface, following the repository
worker-extension example. The extension sets `load_rng=True` through Bridge's
existing pre-load hook. It requires a real resume, rejects fine-tuning fallback,
and hashes optimizer moments **and FP32 master parameters**, rather than the
distributed optimizer's common-only state dictionary. Acquisition controller,
scoring, training, and checkpoint code remain byte-identical.

## Qualification

The fixed source is EOS job `474952005`, learner version 7, capsule manifest
`184004bf...c4284c7`. Two independent processes load the same checkpoint and
same four comparator groups. Each authenticates four withheld groups,
evaluates pre-update loss without changing training state, and executes one
comparator update. Observed Adam counters must advance 7 to 8; scheduler
sample counters must advance 224 to 256. Complete before/after state hashes
must agree between duplicate replays. No Shield update, post-update held-out
loss, or historical treatment contrast is computed.

## Transport and verification

The complete checkpoint is 14,830,649,614 bytes. Stream individual authenticated
members directly from EOS into `/workspace/m4-duplicate-update-input-capsule`,
outside `{assets_dir}`. Only comparator and held-out tensors are fetched;
Shield-only tensors are excluded. Runtime credentials are read from the existing
CI job environment, never embedded. Redirects fail closed instead of forwarding
credentials. Immutable results/logs are uploaded; no checkpoint is copied to
the downstream collector.

The exact 10 MB base source plus replay overlay was extracted in a fresh
temporary directory. Eight dependency-free tests passed there, checking state
and outcome violations, observed counters, unsafe paths, streaming checksums,
overwrite refusal, and exclusion of Shield-only tensors. Embedded Python,
expanded Bash, Ruff, and compilation checks pass. An initial local validator
used an oversized command argument; it was corrected to stdin and the clean-room
test then passed. No launcher invocation was consumed by that local correction.

The local Mac cannot execute the CUDA/Ray replay. The qualification workload
therefore runs the same contract tests and a real torch training-state-digest
smoke check before downloading the checkpoint and running GPU qualification.
The package receipt records static checks only, not a qualification pass.

## Submission boundary

One new workload, two sequential duplicate comparator replays, four-hour
allocated limit, no queue deadline, and exactly one `runllm.py --no_wait`
invocation. The guard is created exclusively before invocation; uncertain or
failed submission cannot auto-retry. Qualification must terminal-authenticate
before the fresh 20-seed acquisition is packaged or launched.

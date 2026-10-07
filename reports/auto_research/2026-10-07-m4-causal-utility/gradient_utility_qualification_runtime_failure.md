# Gradient-utility qualification runtime failure

The brace-corrected qualification created upstream pipeline `72130197`, child
pipeline `72130279`, EOS job `473763069`, and Slurm job `6186859`. The workload
allocated on `eos0493`. The source-extraction, pinned-Megatron, configuration,
gradient-utility unit-test, and controller-test gates passed. The workload then
stopped before the training command with exit code 5 at the split-step test.

The terminal output establishes the cause: `megatron.bridge` was unavailable,
so `tests/unit/models/policy/test_megatron_split_state.py` was skipped at
collection and pytest reported `Running 0 items in this shard`. Pytest exit code
5 stopped the shell under `set -e`. The path-limited source archive intentionally
omitted `3rdparty/`, but the replacement package did not restore the separate
pinned Megatron-Bridge source archive used by prior authenticated M4 packages.

This was a packaging failure, not a gradient measurement. No training command,
optimizer step, scheduler step, learner-version advance, gradient ledger,
qualification result, or scientific outcome was produced. The empty metrics
directory and absence of the run log, gradient ledger, result JSON, and artifact
inventory corroborate that boundary.

The narrow repair restores the already preserved Megatron-Bridge archive with
SHA-256 `1429945d1d50045900e40314fa283fa5f66484e945aad4920da137d7ad2c9313`
and prepends its `src` directory plus the pinned Megatron-LM directory to
`PYTHONPATH`. It retains the split-step test; it does not convert the missing
dependency into an allowed skip. Source commit, audit protocol, model, workload,
seeds, two-group limit, zero-update gates, runtime limit, and queue behavior are
unchanged.

Authenticated terminal records:

- Main-job artifact ZIP SHA-256:
  `9191976833fba6e3270ae0beb0e4d787da5df22f81f988315ca631b7e1643f6b`
- Logs-after artifact ZIP SHA-256:
  `81a0f9717fb57d68493a64ba6cce5a67b302199b996dd9c043042c8d160f5632`
- Workload output SHA-256:
  `ecf1a92b69f4e6f1b20645a707d99cf2087922ce610796eac1ebd7fa2dac8563`
- Slurm output SHA-256:
  `163a0a046d4685d0b2ec6ed03a3fce6bdc317432d4093569d7a98aa02cd226aa`


# Gradient-utility qualification mcore test-runner failure

The mcore-interpreter-repaired qualification created upstream pipeline
`72143534`, child pipeline `72143681`, EOS job `473862015`, and Slurm job
`6187450`. Safe extraction, configuration authentication, all seven
gradient-summary tests, and all three controller configuration tests passed.
The baked Megatron worker interpreter then successfully imported both
`transformer_engine` and `megatron.bridge`, proving that the preceding
interpreter repair selected the intended mcore environment.

The next command failed before collection because that production worker
environment intentionally omits the test-only `pytest` package:

```text
/opt/ray_venvs/.../MegatronPolicyWorker/bin/python: No module named pytest
```

The training command did not start. No optimizer step, scheduler step,
learner-version advance, gradient ledger, qualification result, or scientific
outcome was produced. The artifact bundle contains no run log, gradient
ledger, result JSON, or artifact inventory.

The narrow replacement copies only pytest and its pure-Python runner
dependencies from the image's base test environment into an isolated shim
directory. It prepends that shim only for the split-state test process, keeps
the mcore interpreter and its compiled Torch/Transformer Engine stack, disables
third-party pytest plugin autoload, and selects the repository's mcore tests
explicitly with `--mcore-only`. It performs no installation and no network
access. Source commit, protocol, model, workload, seeds, two-group limit,
zero-update gates, runtime limit, and queue behavior remain unchanged.

Authenticated terminal records:

- Main-job artifact ZIP SHA-256:
  `ce7a11837c749469e3b2aeb64cbe2dbfbc08316070c6433569c14aec5ab2234c`
- Logs-after artifact ZIP SHA-256:
  `ebe437394ae1d2e689d161bf4975b823a02eeffc76f188450d6994e003809a00`
- Workload output SHA-256:
  `8fa453b54fa1554b649fba2ae8caf1d6b27deb5cf5c4763bf94332701f2e1888`
- Slurm output SHA-256:
  `423126b5d83d4c16dac7bfbd52fd677d330fdff8c785a1836de6529f0ff341b9`

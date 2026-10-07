# Gradient-utility qualification mcore-interpreter failure

The dependency-repaired qualification created upstream pipeline `72132732`,
child pipeline `72132864`, EOS job `473783206`, and Slurm job `6187011`. The
workload allocated on `eos0571`. Safe extraction, the frozen configuration
gate, all seven gradient-summary tests, and all three controller configuration
tests passed. The workload then stopped before the training command with exit
code 5 at the split-state test.

The terminal output identifies the exact boundary. The launcher invoked the
`mcore`-marked split-state test with `/opt/nemo_rl_venv/bin/python`, the base
driver interpreter. That interpreter does not contain `transformer_engine`.
Consequently `pytest.importorskip("megatron.bridge")` skipped the test at
collection, pytest collected zero runnable items, and returned exit code 5.
Adding the pinned Megatron-Bridge source repaired source discovery but could
not add the compiled Transformer Engine dependency to the base interpreter.

This was a pre-acquisition environment-selection failure. The training command
did not start. No optimizer step, scheduler step, learner-version advance,
gradient ledger, qualification result, or scientific outcome was produced.
The artifact bundle contains only JET logs; the run log, gradient ledger,
result JSON, and artifact inventory are absent.

The image already contains the correct baked mcore worker environment at
`/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python`.
Authenticated job `470731333` on the same pinned image loaded Transformer
Engine from that environment, loaded Megatron-Bridge from `/opt/nemo-rl`, and
completed 64 Llama-3.2-1B learner updates. The narrow replacement therefore
runs the split-state test with that mcore interpreter and restores the proven
training environment (`PYTHONPATH` equal to the replacement repo). It does not
change source commit, protocol, model, workload, seeds, two-group limit,
zero-update gates, runtime limit, or queue behavior.

Authenticated terminal records:

- Main-job artifact ZIP SHA-256:
  `83b5fb83516ecda4ba9b19169ae120aa8fe659bc64f4c8e9c07431681f5537ea`
- Logs-after artifact ZIP SHA-256:
  `85afea9adf10d1922982cf9f922516cf281891c16c08456f75c379a612403894`
- Workload output SHA-256:
  `7206cfa223024561b2be8317488f6dd6c25d5de449e378c52dad0068b1e483ca`
- Slurm output SHA-256:
  `648545f9dfc570134d2806bbfcd796b9cc1d9552b32df673e953153930f92d84`

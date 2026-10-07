# M4 gradient-utility hash-v2 qualification: terminal report

Status: **PASS_RUNTIME_QUALIFICATION_HASH_V2**

The source-bound CountSketch repair passed in upstream pipeline `72213451`,
child pipeline `72213641`, EOS job `474384649`, and Slurm job `6190430`.
Every preflight test and no-update runtime invariant passed. The two exact
gradient norms reproduced the prior qualification values, isolating the repair
to sketch hashing.

Sketch fidelity improved from the v1 warning range of 12–16% relative error to:

| seed | median relative error | p95 relative error | frozen gate |
|---:|---:|---:|---|
| 20261019 | 0.001463 | 0.001677 | pass |
| 20261021 | 0.001760 | 0.002148 | pass |

Both medians are far below the frozen `0.10` maximum, and both p95 values are
far below `0.25`. The seed-partition unit test passed in the runtime image.
Both groups acknowledged abort; optimizer steps, scheduler steps,
learner-version advances, and `finish_train_step` calls were zero; and the
canonical parameter hash was unchanged.

All six compact files reproduce the runtime-generated SHA-256 inventory. This
closes the repaired instrument qualification without opening any 256-group
construct outcome. The contingent scientific acquisition may now start under
the unchanged frozen audit protocol.

Authenticated bundles:

- Main artifact ZIP SHA-256:
  `ad781960959239a6dbc2cf0f96443ab1aac02c194fbb9f604a07768215e62820`
- Logs-after artifact ZIP SHA-256:
  `ff940df7dd0016f0e720c12c4164880874da35980767b3ec6bd13d70db5b2893`
- Formal result SHA-256:
  `83680b74e4a1eb253d53d9ca5155d469a030e71c5973b87f31c698c72ae0237f`
- Gradient ledger SHA-256:
  `5aab0d8328e541e565896d7f40324d2e04c8c976d04b4bc1d7e28dd7994e16d4`
- Artifact inventory SHA-256:
  `cf874fa8811e374f7cb4d896150c5f1850160c7c89a078fe93cc259d13ebf2e1`

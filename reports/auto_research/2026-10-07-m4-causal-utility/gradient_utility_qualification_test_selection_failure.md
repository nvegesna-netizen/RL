# Gradient-utility qualification MCore test-selection failure

The `py`-dependency-repaired qualification created upstream pipeline
`72200749`, child pipeline `72200849`, EOS job `474287121`, and Slurm job
`6189905`. Safe extraction, configuration authentication, all seven
gradient-summary tests, all three controller configuration tests, the MCore
import gate, and isolated pytest startup passed.

The MCore-interpreter test command collected the split-state test module but
selected no tests:

```text
37 deselected, 18 warnings in 13.15s
```

Pytest returned exit code `5`. The selector used implementation-operation
names (`abort_train_step` and `begin_train_step`), while the collected tests
are owned by the `TestAbort` and `TestBegin` classes and do not contain those
operation names in their node IDs.

The training command did not start. No optimizer step, scheduler step,
learner-version advance, gradient ledger, qualification result, or scientific
outcome was produced. This is a pre-acquisition test-selection failure.

The narrow replacement names the frozen test nodes directly: the complete
`TestBegin` and `TestAbort` classes plus
`TestGradSyncFuncLifecycle::test_begin_abort_round_trip`. It changes no source,
model, workload, seeds, audit behavior, two-group limit, zero-update gate,
runtime limit, or queue behavior.

Authenticated terminal records:

- Main-job artifact ZIP SHA-256:
  `80e284dab3e4508ea8b635b735aa9eb4bd747eaf728debf5fc70e94641ac4126`
- Logs-after artifact ZIP SHA-256:
  `f1d00c472099dfb81de2f7257eaf5b985f866cafb93ffc926fb0ce6e0569561d`
- Workload output SHA-256:
  `17b17068149a74d80d297d43f70ef57fc8abccfcc49b3f52c4d721cb71ed6288`
- Slurm output SHA-256:
  `c2f132d63e3aac5dacbd3a5413a66bdc9126fe2e33183ac84b714c8b2f64a082`

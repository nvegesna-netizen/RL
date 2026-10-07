# M4 gradient-utility runtime qualification: terminal report

Status: **PASS_RUNTIME_QUALIFICATION**

Upstream pipeline `72206571`, child pipeline `72207068`, all three EOS jobs,
and Slurm job `6190180` completed successfully. The source, repair chain, and
submission manifest authenticated before execution. Preflight passed seven
gradient-summary tests, three controller configuration tests, twelve MCore
begin/abort tests, and two offline analyzer tests.

The frozen qualification produced exactly two unique prompt groups with eight
siblings each. Their exact gradient L2 norms were `8.783870995975551` and
`6.499511124023895`. Each group produced both required 16,384-bin sketches and
acknowledged the abort path. The terminal record reports:

- `finish_train_step_calls = 0`
- `optimizer_steps = 0`
- `scheduler_steps = 0`
- `learner_version = 0`
- `parameter_hash_unchanged = true`

The formal result JSON is `PASS_RUNTIME_QUALIFICATION`. All six files in the
runtime-generated artifact inventory independently reproduce their recorded
SHA-256 hashes. Corrected observer duty was `0.0031876195848652687`.

This closes the implementation and runtime-qualification gate only. It does
not classify M4's gradient-utility construct, evaluate scheduler quality,
measure terminal training quality, or support production-readiness claims. The
frozen 256-group scientific acquisition has not started.

Authenticated bundles:

- Main artifact ZIP SHA-256:
  `93dcc1362b9d951ea57ed0d5f09008e52a2717acc861994515c75bebaba487ee`
- Logs-after artifact ZIP SHA-256:
  `f926319df461cc92a026561f010788ce8c11633c94c652d72145deb3f3b1e2c6`
- Gradient ledger SHA-256:
  `a372babaa515f7bb4bbc6773ea85a3ad3f83ccc3e458bf1cb57b7d4f1e29b348`
- Formal result SHA-256:
  `4ee893d47f4f4eef37f63538866b3f5504225a144877593afc9ba94c100207c8`
- Artifact inventory SHA-256:
  `276e71377990991c8e984fa4acbb156d0d25bafd5520c9c42c86468ac26ec0c5`

The next scientific step is a separately opened 256-group acquisition under
the frozen audit protocol, followed by the preregistered cross-fitted and
bootstrap analysis. It must not be described as a retry of qualification.

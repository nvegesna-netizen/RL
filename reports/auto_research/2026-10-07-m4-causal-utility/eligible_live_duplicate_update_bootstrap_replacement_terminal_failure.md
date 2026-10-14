# Bootstrap replacement: incomplete dependency packaging

2026-10-14. Parent `73836981`, child `73837493`, EOS job `487356704`.

All 10 CPU tests passed, including the new fresh-process Core path and worker
import-order regressions. The original `megatron.core` lookup failure is
resolved. The full worker smoke import then failed at the next dependency:

```text
nemo_rl/models/policy/workers/megatron_policy_worker.py:29
from megatron.bridge.training.checkpointing import (...)
ModuleNotFoundError: No module named 'megatron.bridge'
```

This is still a package/environment failure, before checkpoint retrieval,
restoration, or a training update. It produces no causal-utility observation.
The preceding bootstrap repair was incomplete: it verified Core source
resolution, not the complete worker dependency graph. The full EOS import
check correctly prevented progression, but this omission must be addressed
before another submission.

## Source-level evidence

The frozen base source archive contains only the Bridge submodule directory,
not its implementation. The package separately extracts Core, but no Bridge
archive. The standalone worker smoke check runs in
`/opt/nemo_rl_venv/bin/python`; Bridge is unavailable on its effective import
path. The new package also explicitly forces system Python for policy actors
through `NEMO_RL_PY_EXECUTABLES_SYSTEM=1`. That override selects system rather
than the normal MCORE executable; it must be justified and checked when
reconstructing the complete driver/actor environment.

There is a version-matched Bridge source already preserved locally:

- Base source commit `11143be...` pins Bridge gitlink
  `573e088c9c6740082c39744e03dc5b009e730ed4`.
- `m4-oars-preflight-megatron-bridge.tar.gz` has the same commit in its Git
  archive PAX header, 1,060 members, and SHA-256
  `1429945d1d50045900e40314fa283fa5f66484e945aad4920da137d7ad2c9313`.

Therefore the source omission can be repaired without choosing a new Bridge
version. This does not yet establish that all transitive dependencies are
available or that the full worker can import; those remain execution gates.

## Terminal evidence

Slurm `6250803` on `eos0044` queued for `03:50:06`, then ran for `00:02:54`;
the workload step lasted `00:02:32`. It exited `1:0` with state `FAILED`, not
`TIMEOUT`. The roughly four-hour GitLab duration includes pending allocation.

The test log is 1,531 bytes and is byte-identical in EOS and logs-after copies,
SHA-256 `083b18bfc69bcb0c0a6b9081ea9d36f40c754c4357c641f04bb7bc88f713e059`.
The EOS trace is 40,815 bytes, SHA-256
`db6cb173d62280bbf3900535f21beddb1e4d4347f367c02d0664cde3fb5e4c70`.
These are direct-retrieval hashes, not a workload checksum ledger. Transport,
replay, and qualification result artifacts are absent, consistent with the
trace stopping before transport invocation.

## Required next action

Reconstruct and pin the complete worker environment, using the matched Bridge
archive and the original locked dependencies. Check the intended driver and
actor interpreters and require a full worker import in each, not merely a
`find_spec` path check. Keep the full import preflight before large checkpoint
retrieval. Preserve the same comparator replay, checkpoint, state counters,
hash comparisons, outcome embargo, four-hour allocation limit, and no queue
deadline. A newly authorized replacement needs a distinct package and guard.

This diagnosis changes no source and launches nothing. Both consumed guards
and original receipts remain intact; the 20-seed acquisition remains unlaunched.

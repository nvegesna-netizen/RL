# Duplicate-update qualification: setup failure

2026-10-14. Parent `73740465`, child `73740815`, EOS job `486620007`.

The workload failed before retrieving the historical checkpoint or running
either comparator replay. The source/archive checks and all eight CPU contract
tests passed. The next training-state-digest smoke check failed while importing
the qualification worker:

```text
research/m4_eligible_replay/worker.py, line 12
from megatron.core import tensor_parallel
ModuleNotFoundError: No module named 'megatron'
```

## Cause

The package's smoke check imports `research.m4_eligible_replay.worker` directly
in the baked Python interpreter. That module imports `megatron.core` before its
first NeMo-RL import. NeMo-RL initialization (`nemo_rl/__init__.py`, lines 37--46)
adds the extracted Megatron-LM source to `sys.path`, but has not run yet in this
fresh process. The pinned archive contains
`Megatron-LM/megatron/core/__init__.py`; this is an initialization-order defect,
not evidence that the checkpoint or replay action is invalid.

The earlier successful restoration runner imported NeMo-RL modules first and
did not directly import this new worker for a standalone smoke check. The
local clean-room tests were dependency-free and compilation-only for the CUDA
worker, so they did not catch its fresh-process import ordering. The EOS smoke
check caught it before any expensive checkpoint retrieval or update.

## Timing and preserved evidence

Slurm job `6248469` on `eos0432` queued for `06:55:25`, then ran for `00:02:44`.
The workload step was `00:02:23`. Terminal state is `FAILED`, exit `1:0`, not
`TIMEOUT`. The submitted command has exactly one `--time 0:14400` flag and no
queue deadline flag. GitLab's roughly seven-hour job duration includes waiting
for allocation; it does not represent seven hours of GPU execution.

The 1,206-byte test log is byte-identical in EOS and logs-after artifacts, with
SHA-256 `d748aeba6ee5b3dd333127499d2f2cb830a8f6f050b64504537446d638fb8704`.
The direct EOS trace is 35,209 bytes, SHA-256
`299945765a5b398d642169f15720f4efc4f6ba8b28e16fcaa8e691011fd9636a`.
These are independently computed retrieval hashes; the workload did not reach
its checksum-writing stage. Transport and replay logs/results return 404,
consistent with the trace stopping before transport invocation.

## Next repair

Initialize the NeMo-RL bundled-source path before standalone Megatron worker
imports, and add a fresh-process import regression rather than relying on a
process that already imported NeMo-RL. Preserve the same checkpoint, comparator,
state checks, duplicate replay, outcome embargo, and execution limits. Verify
the full worker import in the exact baked EOS environment before downloading
the checkpoint; check the actor environment as well rather than assuming the
driver smoke check covers it.

No code repair or replacement was launched in this diagnosis. The original
manifest, receipt, and consumed guard remain unchanged. Any replacement needs
a distinct frozen package and guard, not reuse of the failed submission. The
fresh 20-seed acquisition remains unlaunched.

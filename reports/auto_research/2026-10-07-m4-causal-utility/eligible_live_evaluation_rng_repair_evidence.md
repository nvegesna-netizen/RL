# Evidence for an evaluation-local Megatron RNG repair

2026-10-15. Evidence gathering only. No runtime source edits, new manifest,
guard reset, commit, push, dependency installation, GPU allocation or launch.
Prior paper edits remain untouched.

## What the terminal records establish

Reauthenticated `replay-a-state-comparison.json` from EOS `490015455` and
collector `490015463`. Both match SHA-256
`3e4288238103c89f40f5b9a7481ed6e907262c4c7314a8450efaf91d207fb846`.
Independently compared the embedded snapshots, rather than trusting their
precomputed difference lists:

- Initial and idle snapshots are identical.
- Evaluation changes only `rng_component_sha256.tracker` and the combined
  `rng_sha256` that includes it.
- Parameter/optimizer/scheduler fingerprints, counters and restoration flags
  are identical. Python, NumPy, Torch CPU and default CUDA RNG hashes match.
- Comparator training has not been called; replay B has not started.

The tracker fingerprint changes from `221bcbf2...7360b46` to
`c7b920a3...afe9fc7`. These hashes identify a change in tracked contents, not
the exact stream name or internal random operation. They do not establish
that model parameters changed or that training was attempted.

## Exact pinned-source provenance

Inspected the archived source actually packaged, not a newer checkout:

| Source member | SHA-256 |
| --- | --- |
| Megatron-LM `megatron/core/tensor_parallel/random.py` | `c2bc8b53e27f0a883eff360d5ba92071a6a27905ea03035426f4c071d3d036e4` |
| NeMo `nemo_rl/data_plane/worker_mixin.py` | `0feef374433576a17d02d51a01569ca224590af0f89e18393be17e9af2d4623a` |
| NeMo `nemo_rl/models/policy/workers/megatron_policy_worker.py` | `160362a38ebe73eec5cfbc76d84355ea4c32f5c8c7aac8be8f4a7c13b461b7e9` |

Megatron archive SHA-256:
`98d98920c0fea3d8ae1216a485dc9b5aa4fc966e469435ad61f2bae456de80d2`.
NeMo base archive SHA-256:
`ff8cb7f09b64b6ff5d781c12c52d566c8c8821ebab559d569e322ec2fabe691f`.

### Correct worker hook

Pinned `TQWorkerMixin.train_presharded`, line 455, fetches and prepares data,
then calls `self.train(data, loss_fn=..., eval_mode=..., gbs=..., mbs=...)`.
Dynamic dispatch therefore reaches a `train` override on our existing custom
`EligibleLiveReplayWorker`. Its exact inherited signature is:

```python
train(self, data, loss_fn, eval_mode=False, gbs=None, mbs=None,
      check_dim_skip_keys=None)
```

The override must preserve these arguments and forward them unchanged. Branch
on `eval_mode`; normal training must delegate directly without RNG restoration.
Do not alter shared acquisition workers, TQ dispatch, loss implementation or
checkpoint loading. No separate driver-side restore RPC is needed.

Pinned `_fetch` and `_attach_or_repack_pack_metadata` contain no explicit
RNG/tracker/seed calls in their ASTs. That is not proof about every transitive
dependency. The outer full-state equality gate must remain to detect changes
outside the proposed hook.

The pinned worker's evaluation branch uses `torch.no_grad()` and `model.eval()`.
Explicit optimizer and scheduler steps are guarded by `not eval_mode`. The
method synchronizes CUDA at lines 758 and 989 on its normal execution path.
The repair should wrap the complete inherited evaluation call, not an interior
attention subroutine or a driver loss calculation.

### Why ordinary RNG forking is not the repair

Pinned `CudaRNGStatesTracker.get_states()` at line 271 copies the dictionary
but retains references to the state values. A shallow copy is not an isolated
backup against in-place mutation.

`set_states(states)` at line 279 installs a state dictionary and marks the
tracker initialized. It does not reseed the model or change the default CUDA
generator. Use it only with a real initialized tracker and cloned supported
state values; never call `reset()` to emulate restoration.

`fork()` at line 311 deliberately updates the named tracked state on exit,
then restores the previously active default CUDA generator. This matches the
observed combination of changed tracker and unchanged default CUDA RNG.
Wrapping evaluation in another `tracker.fork()` would retain the wrong
semantics. A Torch-only RNG context also does not establish restoration of
this separate Megatron tracker.

## Independent CPU semantics experiment

Executed the actual pinned `CudaRNGStatesTracker` class AST with stand-in
CPU/CUDA state primitives. The candidate save/restore context existed only
in memory; it was not installed in the worker. Eight assertions passed:

1. Actual pinned `fork()` preserves the stand-in default CUDA generator while
   advancing its tracked generator.
2. A `get_states()` shallow-copy alias changes when the tracked value is
   modified in place.
3. Saving cloned values and restoring through `set_states()` preserves state
   across a normal evaluation exit.
4. Repeated isolated evaluations return the same stand-in result.
5. An evaluation exception still restores state and preserves the original
   exception object.
6. An unwrapped training path still advances tracked state.
7. Cloning prevents the saved backup from sharing in-place mutations.
8. The ordinary tracker remains initialized in its default context afterward.

These tests validate control flow and the pinned class semantics. They are
not real Torch tensor, Transformer Engine, CUDA graph, numerical-loss or
optimizer-update tests. No GPU or cluster allocation was used.

## Smallest justified production repair

1. Add an evaluation-only `train` override to the custom qualification worker,
   matching the inherited signature. Bypass isolation entirely for training.
2. Obtain the already initialized tracker inside that worker. Snapshot its
   named states using independent clones of the supported state values.
3. Call the inherited evaluation normally. Preserve its returned metrics and
   exception semantics; do not modify loss values or model/optimizer state.
4. Restore the saved tracked state through the verified API in `finally`,
   including exceptional exits. Do not manually reseed or reset any generator.
   Do not replace a primary evaluation exception with a cleanup exception;
   attach restoration failure information and fail closed.
5. Keep all per-field diagnostic records and the original complete
   `before != before_update` gate. Do not remove tracker/RNG fingerprints,
   change tolerances, or copy the earlier hash into the later record.
6. Preserve all other RNG-domain checks. The evidence supports repairing the
   affected tracker; it does not justify globally resetting unrelated RNGs
   and potentially hiding a new source of mutation.

### Type and mode checks before claiming compatibility

Pinned `get_cuda_rng_tracker()` returns the already selected global tracker;
its default arguments do not prove which tracker was initialized earlier.
`initialize_rng_tracker()` supports Core, Transformer Engine, inference and
graph-safe variants. The pinned TE wrapper inherits its get-state behavior
from the installed Transformer Engine package. Do not assume all variants
have tensor-valued state or identical cloning semantics.

The repair should record the actual worker tracker class and relevant mode,
verify supported state-value types and a clone/restore round trip, and fail
loudly for unsupported cases. Do not turn on another tracker, force-reset it,
disable graphs or silently swap implementations just to make the patch run.
The terminal snapshots were serializable by the existing state digester;
that alone is not a complete runtime tracker-class inventory.

## Verification before any later launch

- Unit tests of the actual override: normal and exceptional evaluation,
  argument/return forwarding, no aliasing, and direct training passthrough.
- Use mutable tensor-like tests as well as real tensor checks in the existing
  EOS preflight. A shallow-copy mock alone is insufficient.
- Verify the actual TQ shim dispatches into the override; do not test only an
  unused context-manager helper.
- Check restoration failure remains fatal and retains the original exception.
- Retain the existing mutation-negative tests for parameter, optimizer,
  scheduler and all RNG fields. Unexpected tracker-key/mode changes must not
  be silently accepted as generic compatibility.
- Clean-room extract and test the exact package, verify hashes/schema/shell,
  keep backend and protocol bytes pinned, and preserve every consumed guard.
- Only a later authorized EOS qualification can establish real before/after
  equality and duplicate one-update reproducibility. The comparator update
  must still change parameters and advance its counters; RNG isolation must
  not freeze the actual training operation.

The current investigation does not implement this repair or authorize another
invocation. It establishes the hook, API behavior, aliasing risk, scope and
tests needed to implement it without weakening the scientific contract.

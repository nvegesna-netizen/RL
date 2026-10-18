# Minimal-observer bridge qualification

2026-10-18. User authorized the next step with “do it.” This is one engineering
qualification on the excluded historical capsule, not a new hypothesis test.
Branch: `codex/m4-minimal-observer-qualification`, parent `2cf6600c4`.

## Why this check

The authenticated strict instrumented run reproduced both comparator updates
exactly, including gradients and full post-update state. Deep observation
copies GPU tensors and explicitly synchronizes gradients/optimizer snapshots.
The passing CPU traced/control smoke does not establish that removing those
observers leaves the real Megatron GPU update unchanged.

The bridge therefore repeats the same strict deterministic comparator update
in two fresh processes without deep hot-path observation. It requires exact
duplicate equality and exact agreement with the accepted instrumented
pre-/post-update state. There is no approximate numerical tolerance for state
hashes, new action selection, historical treatment contrast or quality outcome.

## Frozen boundaries

- Original duplicate protocol remains byte-identical. A new bridge protocol
  hash-binds the authenticated strict terminal reference.
- Same base source, Core/Bridge archives, image, model, BF16, packing, sample
  groups, loss, optimizer, checkpoint, evaluation RNG isolation and strict
  Torch/model/environment controls. No backend source or recipe edits.
- Builder retains the previous test suite but removes the actual
  `--training-diagnostics` invocation and trace-comparison command. The runner
  explicitly selects minimal mode; it rejects missing strict mode, missing
  boundary directory and concurrent deep diagnostics before Ray setup.
- The actual worker rejects deep trace configuration in minimal mode and
  delegates training through the already existing untraced superclass path.
  No new model forward hook, microbatch wrapper, gradient hash or optimizer-step
  wrapper is installed. Runtime metadata and full-state hashes remain outside
  the update; this is minimal observation, not zero observation.
- Boundary RPC checks actual worker modes and records source identity before
  evaluation/update and after the update. Both boundaries and both processes
  must agree with reference runtime metadata, except the intentionally changed
  custom worker source hash. That current hash must match the packaged file;
  every other runtime field and module identity remains compared exactly.
- Original complete duplicate-state validator is unchanged. An additional
  bridge validator requires exact reference state agreement and forbids deep
  trace artifacts. Counter increments remain 7 to 8 and 224 to 256. Parameters
  must actually change. Held-out files stay unchanged, with no post-update
  held-out loss. The original qualification result is persisted only after
  the cross-reference check passes.

Final manifest SHA-256:
`de2ee7b5c1b4437b1d6ed53b9f930a3bf96dca471f31aee780aea3844fab238e`.
Bridge protocol SHA-256:
`bca546a19be0470aef206b11afa5637053dab3744e0a4db42878a4e5c323b7b8`.

## Verification

The exact extracted final package passes 106 tests, zero skipped locally.
Sixteen shell tests pass. These include actual worker no-trace dispatch and
mode rejection, actual runner pre-setup rejection, duplicate/reference state
mutation, source-hash authentication and runtime downgrade tests. Actual pinned
Bridge/Core checks and all retained restore/RNG/trace contract tests remain.
Independent rebuilds are byte-identical. JET schema, Bash and embedded Python
syntax, extracted compile checks, Ruff and scoped whitespace checks pass.

AST comparison against the accepted parent confirms the scientific `_run`
body is unchanged after removing only the new minimal-mode config and boundary
records. The native worker train dispatch is unchanged after removing the
additional minimal-mode guard. Contracts, deterministic controls, transport,
preflight, Ray startup and RNG helpers remain unchanged. All eight consumed
guards are preserved and hash-bound. Source/package/authorization/config must
be committed before the new submitter accepts them.

Local testing cannot establish the actual GPU bridge result. The real driver
and isolated actor preflight remains mandatory before capsule transport. Its
tiny observer/control smoke is a preflight check, not deep observation of the
qualification update.

## Execution and interpretation

One new exclusive invocation guard; source rotated `setup_jet.sh` before
launch; use `runllm.py --no_wait`, 14400 allocated seconds, no client queue
deadline. No automatic retry, extension, fresh capsule or causal acquisition.
Authenticate parent, child and exact workload job; after terminal status,
authenticate both compact artifact copies before interpreting the gate.

A pass supports observation-insensitive duplicate reproducibility for this
one strict deterministic capsule/update setting. It does not establish
production scheduling performance, a specific original-kernel cause or an M4
quality benefit. A mismatch is an engineering failure to investigate, never a
reason to relax equality or automatically retry. A separate scientific
acquisition remains a later prospective decision.

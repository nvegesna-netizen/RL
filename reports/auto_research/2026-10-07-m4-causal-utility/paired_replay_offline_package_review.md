# Prospective paired replay: offline package and gap audit

2026-10-18. User request: “do it”, following the authenticated minimal-observer
bridge pass. This step prepares and verifies the paired primary-endpoint
execution package offline. It does not launch or unlock scientific acquisition.
Branch: `codex/m4-paired-replay-offline-package`, parent `1775bab3e`.

## What is implemented

The separate `research/m4_paired_replay/` package adds:

- The unchanged twenty-seed v2 assignment/domain, all-twenty terminal
  classification requirement, minimum sixteen valid capsules, no replacements,
  no historical identities, duplicate EOS/capsule rejection and no implicit
  zero outcome for a trajectory without a capsule. An authenticated failure
  before workload creation can be classified without inventing an EOS job.
- Exact recorded action membership, retained in serialized frontier order as
  in the accepted comparator qualification. The scheduler is never rerun.
  Group roles, sample disjointness, canonical paths and checkpoint inventory
  boundaries are checked before selected member transport.
- Direct per-member EOS transport for the checkpoint, the chosen arm's four
  groups and four held-out groups. No collector endpoint or ZIP archive is
  used. Credentials cannot follow redirects. Streaming uses the previously
  qualified bounded size/hash verifier. The full manifest remains hash-bound,
  including unselected frontier metadata.
- A new NeMo adapter using the accepted worker unchanged, full optimizer/RNG
  restoration, strict minimal-observer runtime, the same loss/configuration,
  fixed per-arm tensors and one update. Held-out evaluation occurs before and
  after the update. Full-state fingerprints must show no mutation by either
  evaluation and exactly one real optimizer/scheduler advance.
- Hash-bound comparison to the accepted actor runtime, including actual module
  source hashes. The new adapter is separate from the accepted worker; there
  is no change to the already-qualified runner, worker, deterministic controls,
  scheduler, capture logic, eligibility, model, seeds or scientific endpoint.
- Private endpoint files and logs, and outcome-free compact completion
  receipts. A completion receipt is written only after backend cleanup
  succeeds. Primary errors survive a secondary cleanup failure.
- A two-pass release gate: first authenticate the exact expected arm set,
  provenance, fixed groups, initial state equality, strict runtime, invariant
  receipts and every payload hash; only then deserialize any outcome JSON.
  Missing or invalid arms keep every endpoint closed. After opening, require
  equal pre-update losses and compute the prospectively specified
  Shield change minus comparator change. This code does not perform the final
  bootstrap/sign-flip analysis or claim a result.

Outcome sealing is a logical scientific embargo with file permissions, not
encryption or a security boundary against the file owner/administrator. Python
and native driver stdout/stderr are redirected into a private file, including
metric-bearing exceptions. Node-local Ray logs remain outcome-bearing.
No sealed real-study payload has been generated or opened in this step.

## Verification

The exact new overlay was extracted on top of the accepted minimal-observer
package in a clean temporary directory. All **132 tests pass, zero skipped**:
106 retained qualification tests and 26 new offline tests. Tests exercise
both fixed actions, a synthetic sixteen-pair cohort, numerical contrast
direction, immutable input/evaluation gates, no-op/update counters, private
file permissions, cleanup failures, endpoint-bearing receipt rejection,
payload corruption, missing-arm release prevention and actual process-level
Python/native output redirection. The CLI's backend-failure path is tested in
a subprocess with a deliberately failing fake backend; its synthetic metric
stays out of public output.

The direct-transport tests use an injected synthetic HTTP response provider,
not NVIDIA network access or real checkpoint downloads. The new orchestration
tests use a fake stateful backend. Actual NeMo method selection and held-out/
selected metadata dispatch are inspected in source, but this is not execution
of the real CUDA adapter. Earlier actual EOS duplicate results remain separate
evidence and are not relabeled as a new two-action GPU qualification.

Independent final builds are byte-identical. Ruff, extracted compile checks
and whitespace checks pass. The unchanged v2 design validator again verifies
five predecessor records and nine qualified source files. All nine consumed
submission guards remain unchanged. No new guard or launcher invocation exists.

Frozen offline overlay SHA-256:
`4c73f47fa330a3ab9f3b3ed74c6f40fb079a247d64a66261688c2b6f78971b7e`.
The overlay is intentionally not a JET workload manifest; it contains code
and compact evidence only. The machine-readable runtime amendment and receipt
both retain `gpu_launch_ready=false` / disabled execution.

## Review findings that still prevent acquisition

1. Capture inputs to the pure gate must come from an independently authenticated
   fresh-cohort classification producer. A boolean terminal label or a
   user-supplied evidence hash alone is not source authentication. That producer
   and its raw EOS evidence integration are not implemented by this offline
   primary replay package.
2. The new actual two-action NeMo adapter, post-update evaluation and complete
   private-log routing need pinned-runtime EOS verification before cohort
   launch. Local tests do not prove backend compatibility or GPU noninterference.
3. Qualification used a learner-version-seven checkpoint. The first eligible
   event could occur at version zero, where Adam step entries may be absent.
   The accepted worker fails closed on missing counters; do not exclude early
   decisions or invent zero counters to work around this without qualification.
4. The original secondary endpoint list remains unchanged, but this adapter
   implements only the primary endpoint. Secondary measurements still need
   implementation/qualification or an explicitly approved prospective
   amendment. Do not silently drop them or add gradient probes to the primary
   update hot path.
5. The real logs-after collector and node-local Ray logs need an explicit
   outcome-quarantine/export test. Descriptor redirection is useful evidence,
   not proof that an infrastructure collector cannot expose metrics.
6. Capture/replay EOS workload manifests, immutable execution receipts/lock,
   direct checkpoint transport integration, and new per-identity one-shot
   guards remain a separate execution-preparation stage. This package creates
   no execution-ready lock.

Next complete these offline integration gaps, then freeze the narrowly scoped
engineering EOS check for the new adapter/log boundary. Only after its evidence
authenticates should the fixed fresh acquisition be considered for launch.
Historical treatment outcomes remain excluded throughout.

## Reproduction

From the research worktree, use a new output filename because the builder
refuses to overwrite an existing package:

```bash
PYTHONPATH=. uv run --no-project --offline --python python3 \
  reports/auto_research/2026-10-07-m4-causal-utility/build_paired_replay_offline_package.py \
  --output session/20260914_m4_opportunity_control/paired-offline-new-overlay.json
```

The accepted large source manifest must already be present in the ignored
session directory. The committed builder checks its SHA-256 before extraction.
Large source/checkpoint artifacts and generated offline overlays stay outside
Git; the committed receipt binds their hashes and all new source hashes.

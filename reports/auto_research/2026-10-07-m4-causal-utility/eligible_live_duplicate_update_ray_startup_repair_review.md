# Ray startup repair review

2026-10-15. Implemented the bounded repair from the startup audit. No EOS
launch, new submission guard, checkpoint retrieval or outcome opening.

## Changes

- New standard-library `ray_startup.py` allocates distinct short `/tmp` roots
  for preflight and both replays. Every actual generated root is checked before
  Ray initialization. The budget uses UTF-8 bytes, a ten-digit PID and a
  six-digit collision suffix; planned roots use 92 of Linux's 107 bytes.
- Shared cleanup covers partial initialization and replay setup. A secondary
  shutdown exception is attached to the original failure rather than replacing
  it. A shutdown failure with no original exception stays fatal.
- Preflight now also imports the actual qualification module and records and
  compares Ray/Python versions. The real separate-process import probe remains
  mandatory before transport. Driver-stage metadata is flushed before startup.
- Generated shell checks all stage prefixes before importing heavy modules,
  removes explicit replay root overrides, and retains two separate replay
  process invocations. Package validation rejects unsafe overrides, missing
  layout checks, reordered import/transport gates and altered replay loops.
- Tests exercise the real preflight control flow with fake backend dependencies;
  synthetic pass markers are suppressed so they cannot masquerade as actual
  runtime gate records in test logs.

## Verification

The exact final package at
`session/20260914_m4_opportunity_control/m4-eligible-live-duplicate-update-ray-startup-repair-verified-manifest.json`
has SHA-256
`3124b9b7585aa9af7e888c9c4ce065d099efe51f3a3304558a46a1ae92f185c2`.
Its clean-room extraction passes 29 tests with zero skips. Six additional
submitted-shell regression tests pass. Bash parsing, embedded Python parsing,
module compilation, Ruff and diff checks pass. JET's manifest model accepts
the new identity and unchanged 14400-second allocated limit.

Compared the old `run` function and the new inner `_run` by AST, ignoring source
locations: identical after renaming and removing only the old inner
`ray.shutdown()`, now covered by the outer context. Authentication, seeding,
model/optimizer operations, evaluation, counter checks, returned records and
scientific exclusions are unchanged. Protocol, worker, contracts and transport
also remain byte-identical to the previous committed versions. Backend source
archives and MCORE interpreter are unchanged.

The package receipt binds all overlay hashes. Old manifests/guards are preserved;
the earlier full-environment launcher cannot submit this distinct package.
A future launch step must bind the new manifest and source commit to a separate
one-shot guard rather than reuse a consumed one. This repair record does not
authorize or perform that invocation.

## Limits

These are offline engineering checks, not successful CUDA replay. The actual
MCORE isolated process, optimizer/RNG restoration and duplicate one-update
equality still require EOS qualification. The login-host test that would have
transferred the private helper source was blocked before execution; no bypass
was attempted. The earlier source-free Linux socket boundary test remains
valid but is not a test of the repaired helper inside the container.

No dependency upgrade, queue cap, scientific threshold change, fresh cohort,
paper edit or historical outcome contrast is included.

## Reproduce locally

Run with repository `PYTHONPATH=.` and the available `uv` Python runtime:

```sh
uv run --no-project --offline --python python3 reports/auto_research/2026-10-07-m4-causal-utility/validate_eligible_live_duplicate_update_package.py --full-environment --ray-startup-repair --manifest session/20260914_m4_opportunity_control/m4-eligible-live-duplicate-update-ray-startup-repair-verified-manifest.json
uv run --no-project --offline --python python3 reports/auto_research/2026-10-07-m4-causal-utility/test_eligible_live_duplicate_update_package.py -v
```

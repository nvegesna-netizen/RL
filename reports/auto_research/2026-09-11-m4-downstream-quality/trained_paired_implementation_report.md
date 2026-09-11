# M4 downstream-quality local implementation result

Status: `PASS_LOCAL_IMPLEMENTATION_UNLAUNCHABLE`.

The frozen 16-block, 32-run downstream-quality design is implemented locally.
No model was accessed, no optimizer was initialized, no training occurred, and
no EOS submission was made.

## Implemented boundary

- Two generated-run templates preserve the accepted Qwen3-0.6B/OpenMath
  geometry and materialize all-immediate or equal-mass control/d5 release.
- One deterministic builder produces 16 immediate and 16 mixed-d5 YAML configs
  with unique paths, domains, seeds, and SHA-256 identities.
- Pair normalization proves that each within-block config differs only in its
  release policy, assignment domain, regime label, and isolated output paths.
- The fixed outcome embargo rejects incomplete acquisition gates before opening
  any result file.
- The detached analyzer authenticates all 32 identities, exact 448/448 terminal
  boundaries, prompt manifests, result paths, artifact hashes, pipeline/job
  identities, and resource caps before analysis.
- Primary inference uses 16 mixed-minus-immediate run differences, a paired
  Student interval with 15 degrees of freedom, all 65,536 sign flips, and a
  20,000-draw paired bootstrap. Prompt analyses cannot override the primary.
- The authenticated V8 per-prompt correctness vector was compacted to an
  ignored 102,892-byte covariate artifact with SHA-256
  `86ba0cd85d88e1bac90c27f972effa5038cc08efeae6d3a210922a2bad16cc21`.
  It is sensitivity input only and cannot enter the primary estimator.

## Validation

Six dependency-light tests pass, including negative controls for incomplete
acquisition, mutated training identity, excess aggregate GPU-hours,
unregistered result substitution, and execution-authority mutation. Ruff,
Python syntax, YAML parsing, deterministic rebuilding, and the compact hash
ledger pass. The canonical run-manifest SHA-256 is
`a461e83c7dbccdefa1c5c779062e709f34bfc502639a3e7de663a717fefa93bf`.

Repository-wide pytest collection is still unavailable in this worktree
because its `nemo-gym` workspace member is absent. This is recorded rather than
silently treated as a pass. The implementation tests are dependency-light and
execute directly under the available Python.

## Next boundary

The scientific design and local implementation are ready for acquisition
packaging, not training. The next step is to build and clean-room validate a
credential-free package that performs training, terminal export, conversion,
fixed evaluation, compact result creation, and hash capture for the frozen
schemas. Building that package and launching it are distinct boundaries. No
training or EOS authority exists in this result.

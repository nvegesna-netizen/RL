# Llama lifecycle-derived transport v3 operational amendment

## Scope

This is a prospective package-only successor to the terminal v2 bootstrap
preflight. It does not reopen v1 or v2 and grants no EOS, qualification,
training, acquisition, retry, or extension authority.

The scientific source remains commit
`bdbf2956f3ff46177958a29a64e3b33e4dc6515e` and archive SHA-256
`38b44e4dc97995463f381816f636a59499c3ed84415a3a384165ad412ba43047`.
The estimand, instrument, assignments, seeds, model, workloads, thresholds,
windows, and analysis are unchanged.

## Failure being repaired

V2 required exact equality between the container build fingerprint and a
fingerprint regenerated from a gitless scientific source archive. That archive
cannot emit Git submodule SHAs. It also contains no nested Megatron-LM checkout,
so the repaired `import nemo_rl` then `import megatron` sequence could not be
tested against the intended source path.

## Allowed operational delta

1. Compare `pyproject.toml` and `uv.lock` fingerprints exactly against the
   frozen container fingerprint.
2. Authenticate the container's Megatron-Bridge pin separately and follow its
   gitlink to the exact Megatron-LM commit.
3. Supply a deterministic archive of only that commit's `megatron` source tree
   at the path expected by `nemo_rl`.
4. Require clean-room validation to execute the normalized fingerprint policy,
   import `nemo_rl` immediately before `megatron`, and prove the authenticated
   source path is present in `megatron.__path__`.

Deleting or bypassing dependency checks is forbidden. The dependency archive
is operational packaging, not a scientific-source modification, and must be
bound by its commit, Git tree, byte length, member count, and SHA-256.

## Gate

The only authorized action is local construction and validation of a
non-launchable candidate. Any EOS preflight requires separate explicit user
authorization and a new one-shot guard.

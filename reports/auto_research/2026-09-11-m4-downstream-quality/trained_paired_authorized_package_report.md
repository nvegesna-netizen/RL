# M4 downstream-quality authorized acquisition package

Status: `PASS_32_AUTHORIZED_PACKAGES_NOT_SUBMITTED`.

The user's `I authorize` is frozen as authority for exactly the 32 registered
downstream-quality acquisitions: one attempt per run, 448 learner updates and
terminal version 448, `runllm.py --no_wait`, four wall-hours/eight H100-hours
per run, and no outcome inspection until all runs are terminal and authenticated.
Retry, replacement, extension, qualification, and partial analysis are forbidden.

All 32 local candidates were rebuilt with the new authority payload and distinct
manifest identities. Clean-room validation passed shell syntax, all seven embedded
Python stages, payload identities, dynamic authorization opening, removal of the
old no-execution authority, absence of nested launch commands, and byte-for-byte
deterministic rebuilds. The 571,233,936 bytes of generated manifests remain ignored;
their individual SHA-256 values are recorded in
`trained_paired_authorized_package_validation.json`.

Frozen compact identities:

- Authorization SHA-256:
  `5a64ec577b2acd14f6ddb65e9249abf9718d8655a1f9ff76aea2eed85ae24e30`
- Authorization-bound builder SHA-256:
  `2492dddeadaa504075390dec78b05292de17902a612da69f954b4f1b1f3de024`
- Authorized package validation SHA-256:
  `ebd6449fad9aeea26ce02a0e09a55bd1d2f7488500c2707508a30f1b537f0f83`
- Protocol SHA-256:
  `dbe7a4f7d1938ef43d24e536ccf9dd57110a01bab5e6cb511980218e472ef2c0`
- Run-manifest SHA-256:
  `a461e83c7dbccdefa1c5c779062e709f34bfc502639a3e7de663a717fefa93bf`

No submission was attempted while constructing this package. The next operation
is to commit and push these compact identities, create one exclusive submission
guard bound to that commit and all 32 manifest hashes, then invoke every manifest
once before reading any scientific output.

# Eligible-live restore and held-out evaluation qualification

Date: 2026-10-11

Status: `PASS_RESTORE_AND_EVALUATION_GATE`

The repaired qualification passed its prospectively frozen scientific gate.
Two fresh trainer processes independently restored the same authenticated
checkpoint, requested optimizer and scheduler restoration, and evaluated the
same 32 samples from four disjoint held-out groups. Both produced held-out loss
`0.14215132594108582`; the difference was exactly `0.0`, within the frozen
absolute and relative tolerances of `1e-6`.

The expected parameter hash
`e0765eb28e40eb7308c7c9b2e936179dc12f5da7deb38bdb467dc0e0de075e9e`
matched before and after each evaluation. Both evaluations used evaluation mode
and `torch.no_grad`. Optimizer and scheduler step counts remained zero. All 14
frozen analyzer checks passed.

The compact records were retrieved from the terminal artifact archive and
authenticated against its SHA-256 inventory. Running the preserved analyzer
locally over the preserved protocol and both restore records reproduced the
archived result byte-for-byte with SHA-256
`3fe624a369faa740ad4cb1d5017397c2d112aa5809205d65bd892e9121c6e335`.
The 140,908,811-byte archive remains outside Git and is referenced by SHA-256
`5e8e899d69362f3ee551061575678a3de9f7ff3c92c2e44e8d8c6ac6a718f7d9`.

Pipeline and job states are recorded only as execution provenance; they are not
the scientific endpoint. The scientific endpoint is the authenticated,
independently reproduced gate result above.

This pass establishes that the captured checkpoint can be restored faithfully
and that the frozen held-out endpoint can be evaluated reproducibly without
mutation. It does not estimate causal utility, open either paired arm, claim a
terminal-quality effect, or establish production readiness.

The next stage is to package the separately frozen 20-seed paired acquisition.
That acquisition must preserve the registered arm definitions, one update per
arm, identical pre-update state, no regeneration or action reselection, the
16-of-20 completion gate before outcomes are opened, and the registered paired
analysis. It is a new scientific acquisition, not another qualification retry.

# Llama 3.2 3B no-training preflight plan

Status: `TERMINAL_JOINT_MATERIALITY_NOT_CONFIRMED`.

The first executable boundary is exactly one no-training preflight submitted
with `runllm.py --no_wait`, only after a dedicated authorization record exists.

It may authenticate the pinned source, dependency archive, protocol, and all
six configuration hashes; import the runtime; resolve the two neutral
qualification configs; verify the exact model and tokenizer identities; query
repository metadata/tokenizer access; and confirm one-node/two-GPU topology.

It must fail before model-weight loading, trainer construction, trainer steps,
qualification, acquisition, nested submission, retry, or extension. A green
preflight permits only a later authorization decision for the paired neutral
qualification. It never qualifies the scientific study by itself.

The source-bound local candidate was built and clean-room validated on
2026-09-10. Its SHA-256 is recorded in
`local_preflight_package_receipt.json`; the 15.6 MB candidate remains outside
Git. Because the embedded authorization has `eos_submission_authorized=false`,
the candidate exits at the explicit no-launch boundary before runtime config
resolution or model metadata access if it is submitted accidentally. A later
EOS preflight requires a new, narrowly scoped authorization and a newly built
manifest whose authorization payload permits exactly that one submission.

That authorization was received on 2026-09-10. The launchable manifest was
derived from the validated candidate by changing only its name and embedded
authorization surface, then schema-checked and deterministically rebuilt. Its
hash is recorded in `eos_preflight_package_receipt.json`. This status does not
mean the one-shot submission has been consumed; submission provenance must be
recorded separately after invoking `runllm.py --no_wait`.

The one-shot authority was consumed on 2026-09-10 by parent pipeline
`67234694`. The launcher returned in no-wait mode without polling or
post-processing. This records successful submission only; it is not a green
preflight result and grants no authority for qualification or acquisition.

Terminal authentication identified successful child pipeline `67234855` and
zero-exit workload job `434290055`. The result and hash ledger were identical
in the workload and logs-after artifacts. Both qualification configs resolved,
and Llama 3.2 3B model/tokenizer metadata access passed without model-weight
download or trainer construction. The runtime printed its known nonfatal
container/code fingerprint warning (also present in prior successful Llama
runs); this is preserved in the terminal record rather than silently omitted.
The preflight stage is green, but paired qualification remains a separate,
unauthorized decision boundary.

Paired qualification authorization was received on 2026-09-10. Before any
qualification data existed, `qualification_execution_contract.json` froze the
previously unspecified cell-specific bootstrap seeds. Both 64-step manifests
were built and jointly validated with one-node/two-GPU topology, 4-hour caps,
disabled retriers, one attempt per cell, and acquisition still forbidden.
Status: `PAIRED_QUALIFICATION_SUBMITTED_RESULTS_PENDING`.

The shared one-shot guard was consumed on 2026-09-10. OpenMath parent pipeline
`67239810` and GSM8K parent pipeline `67240055` were both submitted with
`runllm.py --no_wait` before either result was inspected. Both launcher calls
returned zero. No polling, retry, extension, or acquisition was performed.

Terminal inspection found both pipelines failed before training in the same
embedded authorization-evidence block. The manifest builder serialized the
authorization as JSON inside Python, producing lowercase `false`/`true` names;
both jobs stopped at `NameError: name 'false' is not defined` before config
validation, model loading, the training-start marker, lifecycle output, or a
qualification summary. The package validator compiled the block but did not
execute its authorization comparison, so it missed this runtime name error.
Status: `SHARED_PRETRAINING_PACKAGING_FAILURE_NO_QUALIFICATION_DATA`.
The scientific qualification gate was not evaluated. The original two
attempts are consumed, acquisition remains unauthorized, and any repaired pair
requires a new versioned package plus fresh explicit authorization.

The v2 local repair replaced only the workload name, consumed authorization
surface, and defective authorization-evidence block. The builder now embeds a
valid Python representation of the authorization payload. Validation compiled
all six embedded Python blocks per cell, executed the repaired evidence block
against the real frozen payloads, observed its pass marker and explicit
no-launch exit, confirmed the scientific configuration is unchanged, and
reproduced both candidates byte for byte. The compact hash ledger is preserved
in `qualification_packaging_repair_receipt_v2.json`; the large candidates stay
outside Git. No EOS submission occurred. A launchable v2 pair still requires
fresh, explicit authorization and a separately versioned authorization surface.

Fresh explicit EOS authorization was received on 2026-09-10 for exactly one
shared repaired pair. The two launchable v2 manifests were derived from the
validated local candidates by changing only their names and authorization
surfaces. Both passed schema validation, dynamic authorization-evidence
execution, exact-delta comparison, and deterministic rebuild. Each remains a
64-step, one-node/two-GPU qualification with a four-hour cap. Acquisition,
retry, and extension remain unauthorized. Submission provenance is not yet
present in the package receipt, which records the pre-submission state rather
than retroactively treating the package as a launch record.

The shared v2 one-shot guard was consumed on 2026-09-10. OpenMath parent
pipeline `67248517` and GSM8K parent pipeline `67248551` were submitted with
`runllm.py --no_wait`; both launcher calls returned zero. Both submissions were
made before either result was inspected. This is submission success only, not
a qualification result. No polling, acquisition, retry, or extension occurred.

Terminal evidence shows both parent and child pipelines failed, but neither
training failed. OpenMath and GSM8K each completed all 64 trainer steps and
reached trainer version 64 after passing authorization and configuration
checks. Both then hit the same post-training summary defect: the embedded code
subscripted a `JoinedOpportunityAssignment` dataclass as
`row["start_weight_version"]`; its frozen field is `row.start_version`.

The authenticated workload artifacts were recovered and the preregistered
qualification gates were executed offline with only that field-access repair.
Both cells qualify. OpenMath has 847 joined window assignments, a projected
lower 95% assignment count of 6166.67, and projected upper 95% total runtime of
2681.97 seconds. GSM8K has 1390 joined assignments, a projected lower 95%
count of 9833.33, and projected upper 95% runtime of 2151.71 seconds. Every
integrity, lifecycle, duty, assignment-support, and timing-support check passed.
The terminal and offline-recovery records preserve all artifact hashes. The
qualification data remain excluded from any causal estimator. No retraining,
retry, extension, acquisition, or causal estimate occurred.

After both qualifications passed, `acquisition_execution_contract.json` froze
the four causal cells and independent analysis bootstrap seeds before any 3B
causal data existed. Four 448-step local candidates were built for OpenMath r1,
OpenMath r2, GSM8K r1, and GSM8K r2. Clean-room validation confirmed their
source and payload hashes, one-node/two-GPU topology, four-hour caps, schema,
embedded Python and shell syntax, scientific cell identities, dynamic
no-launch guards, and byte-identical rebuilds. Their compact receipt is
`local_acquisition_package_receipt.json`; the 62.6 MB of candidate manifests
remain outside Git. No model-weight access, training, EOS submission, causal
outcome inspection, or acquisition occurred. A launchable four-cell package
requires fresh explicit authorization.

The user subsequently provided that fresh authorization. The separate
`acquisition_authorization.json` binds it to exactly the four frozen 448-step
cells, one attempt each, all submitted before any causal outcome inspection,
using `runllm.py --no_wait` with four-hour per-cell caps. It authorizes the
specified acquisitions but no automatic retry or extension. The scientific
contract and its pre-outcome hashes remain unchanged.

The four authorization-bound manifests were then regenerated from the exact
local candidates. Clean-room validation passed JET schema validation, shell and
embedded-Python syntax, payload ordering and hashes, exact authorization-gate
opening, absence of any nested launch command, and byte-identical rebuilding.
Their compact receipt is `authorized_acquisition_package_receipt.json`; the
62.6 MB manifests remain ignored. No submission or acquisition had occurred at
this provenance boundary.

The joint one-shot guard was then consumed and all four cells were submitted
concurrently with `runllm.py --no_wait` before any causal outcome inspection.
All four launcher calls returned zero. Initial independent verification found
four distinct running parents with created bridges: OpenMath r1 `67263316`,
OpenMath r2 `67263327`, GSM8K r1 `67263343`, and GSM8K r2 `67263328`.
`acquisition_submission_receipt.json` preserves the guard, launcher-log, and
package hashes. The authorization is consumed; no retry or extension exists.

All four parent and child pipelines subsequently completed successfully, with
all 12 child jobs green. Concurrent atomic preservation and authentication
verified GitLab-reported sizes, ZIP integrity, zero exit codes, byte-identical
workload/logs-after evidence, declared hashes, exactly 448 completed steps,
the frozen lifecycle derivation method, and corrected observer duty between
0.000417 and 0.000547. `terminal_authentication.json` is the compact pre-analysis
record. No causal estimator was run before that record and the all-cell analysis
driver were frozen.

The frozen analysis then concluded `MATERIAL` for OpenMath (0.2621, envelope
[0.2215, 0.2994]) and `INCONCLUSIVE` for GSM8K (0.2135, envelope
[0.1872, 0.2398]). Because both co-primary workloads had to be material, the
joint 3B replication criterion was not met. Prespecified 3B-minus-1B contrasts
were negative for both workloads, with simultaneous intervals excluding zero;
the cross-workload difference between those attenuations was inconclusive. The
result supports attenuation in the tested fixed configurations, not a general
scaling law. A deterministic rerun produced identical bytes. The terminal
result is preserved in `terminal_result.json` and explained in
`terminal_report.md`. No retry or extension is authorized.

# Conditional M4 capsule qualification: terminal report

Status: **FAIL_NO_CONTRAST**

The repaired qualification completed its frozen scientific gate. In the tested
Llama-3.2-1B/GSM8K version-zero setting, no shared frontier reached the
preregistered 10% minimum relative predicted-norm gain while preserving
98--102% token service and 100% reward-variance retention. The conditional
one-update bridge therefore stops before treatment, held-out evaluation, or
learning-outcome access.

This is a valid negative qualification, not an implementation failure. The
reward-moment metadata repair worked; archive, dependency, runtime-authority,
and regression checks passed; and all 16 allowed frontiers were evaluated. The
runtime wrapper's generic `RUNTIME_FAILURE` label reflects the deliberate
nonzero exit used when the gate finds no admissible contrast. It is not the
scientific classification.

## Frozen result

The relative predicted-norm gains, reported to the six decimal places emitted
by the runtime, were:

`[0.011847, 0.000000, 0.036869, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]`

Frontier 3 was best at 3.6869%, 6.3131 percentage points below the 10%
threshold at the available log precision. Zero frontiers qualified. No capsule
was written, no held-out groups were collected, no optimizer or scheduler step
occurred, and no learning outcome was opened.

The opportunity ledger contains one header and 192 unique group rows, all at
learner version zero. It records 421,037 valid actor tokens, 158 groups with
positive opportunity and 34 with zero opportunity, binary sibling rewards, and
29 truncated siblings. Group opportunity ranged from 0 to
5,054.875743806362. Corrected observer duty was
0.0005179304927732219 (about 0.0518%) across 192 observations. Only the neutral
release arm existed, and arm-dependent execution was forbidden.

## Integrity and provenance

The source extraction check authenticated 2,956 archive members and 16
symlinks. The Megatron archive and runtime-authority checks passed. The capsule
suite passed 3 tests, and the focused controller suite passed 4 tests with 73
deselected. The runtime artifact inventory independently reproduces every
recorded file hash.

The compact provenance record is in
`conditional_capsule_qualification_terminal_result.json`. Its principal bundle
hashes are:

- Main artifact ZIP SHA-256:
  `455f39ca54e176edf80801a21cf5369e4332f60d07ba8416c6ecaf4167ea8eeb`
- Logs-after artifact ZIP SHA-256:
  `433510e2e3f23b8c8e53fdb11b725e041326026604b053111a7e06b2a8c19387`
- Opportunity ledger SHA-256:
  `f90d4bc1057ed3665f5dc2e9ec278d2b00e114b13f9e730ca8a62b87e9ef2cbb`
- Lifecycle ledger SHA-256:
  `6e85b7833a086220458edd6128cef24995d502c37cb314560e05c6c55345068a`
- Runtime log SHA-256:
  `0993b3006023d2060f7c7f9c14c90068c6e6b97d0bc4bf7ee3b3091575836d9b`

Pipeline and job identifiers are retained only as provenance in the JSON
record. Their status is not a scientific endpoint.

## Interpretation and boundary

This result does not erase the existing evidence that M4 carries predictive
information. It shows that the frozen deployment score did not create a large
enough same-frontier scheduling contrast for the planned causal test under
this strict setting. In other words, predictive association did not translate
into the preregistered degree of local actionability here.

The protocol consequence is to stop this conditional bridge in the tested
setting. Retrying the seed, weakening the threshold, or treating this as an
implementation failure would violate the frozen decision ladder. A future
study, if pursued, must be a new preregistered hypothesis rather than an
extension of this qualification—for example, an observational shadow study of
broader candidate windows or multi-step opportunity accumulation, followed by
a new causal design only if its own prospective gate passes.

One audit limitation must be carried forward: rejected-frontier membership was
not persisted. Exact post hoc frontier-level sensitivity cannot be reconstructed
from the terminal artifact. A chronological regrouping of ledger rows was
checked and rejected because it did not reproduce the runtime's frontier-3
gain; it must not be used as substitute evidence. Any future observational
instrument should persist every frontier proposal and its group membership.

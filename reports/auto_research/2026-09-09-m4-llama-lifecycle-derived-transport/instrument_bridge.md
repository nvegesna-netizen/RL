# Accepted-to-lifecycle instrument bridge

Status: `FROZEN_IMPLEMENTED_PENDING_PREFLIGHT`.

## Invariant scientific quantity

The accepted synchronous instrument and successor lifecycle-derived instrument
must produce the same per-sibling scalar advantages, valid-token counts,
opportunities, group aggregates, and selected-step membership from the same
pre-treatment facts. Only the computation time changes from immediately before
release to after the active run.

| Accepted field | Lifecycle source | Required relation |
| --- | --- | --- |
| `group_id` | group key shared by lifecycle events | exact |
| `sample_ids[i]` | `sibling_done.trajectory_id` | exact identity |
| `sibling_index` | `sibling_done.sibling_idx` | exact 0--7 permutation |
| `sample_id` | `sibling_done.trajectory_id` | equals `sample_ids[i]` |
| `reward` | `sibling_done.reward` | IEEE-754 binary32 equivalent |
| `valid_actor_tokens` | `sibling_done.assistant_tokens` | exact nonnegative integer |
| `truncated` | `sibling_done.truncated` | exact boolean |
| `start_weight_version` | shared lifecycle start version | exact |
| post-release identity check | `group_ready.sample_ids` | exact match when ready |
| completed step | `removed(selected)` plus learner transition | exact membership and version |

The historical final paired qualification established these equalities for all
14,760 observed siblings and reproduced all 64 selected-step mappings. That is
design evidence, not successor qualification data.

## Frozen arithmetic

- estimator: native `GRPOAdvantageEstimator` semantics;
- eight siblings per group;
- binary rewards `{0, 1}`;
- leave-one-out baseline: enabled;
- reward normalization: enabled;
- epsilon: `1e-6` represented in float32;
- rewards, reductions, variance, standard deviation, and scalar advantages:
  IEEE-754 binary32 in sibling-index order;
- coefficient aggregates: float64, matching the accepted ledger definitions;
- valid tokens: assistant-token count, guarded by configuration and equivalence
  tests against the definitive actor mask.

## Causal ordering

The accepted ledger used a computation event between the last sibling and delay
start. The successor instead proves that every source primitive existed before
delay start. A derived row therefore carries source-sequence provenance and a
derivation-method version, not a fabricated controller timestamp. The join must
reject any group whose last required source sequence is not strictly before its
`release_delay_started` sequence.

## Blindness contract

The Q derivation accepts only `sibling_done` projections containing group
identity, start version, sibling identity/index, reward, assistant-token count,
and truncation. Release assignment, delay, nonce, draw, `group_ready`, lifecycle
disposition, learner transitions, and terminal outcome are absent from that
structure. Post-release identity validation and completed-step reconstruction
are separate operations and cannot change Q.

## Fail-closed contract

No imputation is allowed for opportunity. An incomplete group emits no Q, and
the strict join rejects it if its start version is in an opportunity-required
window or it entered a completed learner step. Any unsupported estimator
setting, loss setting, reward, sample masking mode, identity mismatch, duplicate
sibling, mixed start version, ambiguous selected-step mapping, or source event
after treatment start invalidates the artifact.

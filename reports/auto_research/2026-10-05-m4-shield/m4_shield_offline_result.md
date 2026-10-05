# M4-Shield offline development qualification

## Result

`PASS_M4_SHIELD_DEVELOPMENT_QUALIFICATION`

M4-Shield is feasible as a strict safety refinement of the validated OARS-v2
`reward_variance_risk` proposer on the authenticated development corpus. It
improved registered imminent-expiry opportunity while preserving both of the
base proposer's reward-variance objectives exactly. This is retrospective
development evidence, not independent confirmation and not evidence of a
terminal training-quality improvement.

## Frozen policy

The base proposer exhaustively selects four of eight candidates inside the
same 0.98--1.02 FIFO-relative token band used by OARS-v2. Its lexicographic
utility is imminent reward variance, total reward variance, then lower token
count.

The shield considers only four-group batches that:

1. retain at least two of the base proposal's four groups;
2. remain inside the same FIFO-relative token band; and
3. match the base proposal's imminent and total reward-variance sums to an
   absolute tolerance of `1e-12`.

Within that exact utility level set, it maximizes imminent registered L1,
total registered L1, and then lower token count, with a deterministic
group-ID tie break. It acts only for a strict imminent-L1 gain; otherwise it
returns the base proposal exactly. Thus M4 is a Pareto tie-break safety layer,
not a weighted replacement for the validated proposer.

The protocol and all thresholds were frozen before the canonical replay but
after exploratory inspection of the same corpus. The qualification therefore
tests reproducibility and internal feasibility, not out-of-sample efficacy.

## Authenticated scope

- 20 historical arms, 1,280 decisions, and 10,240 candidate rows authenticated
  from the preserved OARS confirmatory archives.
- 64 decisions and 512 candidate rows authenticated from the failed live
  M4-Rescue archive as a runtime-identity and negative-control corpus.
- All 64 live `reward_variance_risk` proposals were reproduced exactly.
- Two full replays produced byte-identical canonical selection records with
  SHA-256
  `5695f1241d5e9ee9d4a139df6a888d479f063707e121965e0997ebbc769eab87`.

## Aggregate findings

| Quantity | Result |
|---|---:|
| Historical base proposals differing from FIFO | 864 / 1,280 |
| Historical M4-Shield interventions | 71 / 1,280 |
| OARS-generated arms with at least one intervention | 10 / 10 |
| FIFO-generated trajectory interventions | 0 / 640 |
| Failed-live steady-state interventions | 0 / 64 |
| Imminent registered-L1 gain over the base proposer | 13,649.7514 |
| Base-proposer imminent-L1 harm relative to FIFO | 9,791.0126 |
| Shielded imminent-L1 harm relative to FIFO | 5,632.3660 |
| Reduction in that observed harm | 42.4741% |
| Harmful base decisions improved | 18 / 29 |
| Maximum change in imminent reward-variance utility | 0.0 |
| Maximum change in total reward-variance utility | 0.0 |

All policy constraints held on every historical and live-negative-control
decision. Every preregistered development gate passed.

## OARS-trajectory detail

| Arm | Base actions vs FIFO | Shield interventions | Imminent-L1 gain | Base harm | Shield harm |
|---|---:|---:|---:|---:|---:|
| p01-oars | 51 | 8 | 1,691.7193 | 1,029.2721 | 254.2446 |
| p02-oars | 48 | 3 | 180.9512 | 40.0148 | 17.6425 |
| p03-oars | 54 | 11 | 1,255.6058 | 588.6962 | 199.5958 |
| p04-oars | 54 | 8 | 800.4928 | 3,139.9609 | 3,015.4110 |
| p05-oars | 58 | 3 | 1,706.5189 | 453.5221 | 0.0000 |
| p06-oars | 47 | 16 | 2,650.6640 | 1,060.7170 | 773.3999 |
| p07-oars | 53 | 4 | 2,220.3822 | 1,968.8981 | 0.0000 |
| p08-oars | 59 | 4 | 804.4517 | 89.8661 | 0.0000 |
| p09-oars | 55 | 3 | 910.7787 | 359.4713 | 359.4713 |
| p10-oars | 50 | 11 | 1,428.1870 | 1,060.5939 | 1,012.6009 |

The effect is heterogeneous. The shield removed all observed base harm in
three arms, reduced it incompletely in six, and did not reduce the one harmful
choice in p09 despite improving M4 opportunity elsewhere in that arm. The
result therefore supports a safety-layer mechanism, not a claim that every
harmful proposal is repaired.

## Negative controls and reachability

The shield made no intervention on any historical FIFO decision and none in
the 64-decision failed-live steady state. This is expected: those trajectories
did not expose an exact reward-utility tie with a strict imminent-M4 gain.
Conversely, all ten OARS-generated trajectories exposed at least one reachable
intervention. The result resolves the earlier M4-Rescue failure mechanistically:
M4 does not bootstrap useful contention from FIFO here, but can refine choices
after an active proposer creates a richer queue state.

## Scientific disposition and next gate

This result adds an implementable systems hypothesis: M4 can serve as an
exact-utility safety layer around an active asynchronous scheduler. It does
not establish that live shield interventions are reachable under a deployed
reward-variance trajectory, that the controller implements the offline rule
exactly, or that the proxy improvement changes terminal model quality.

The next defensible step, if separately authorized, is one outcome-excluded
live systems qualification with `reward_variance_risk` as the actuator and
M4-Shield as the refinement. Its prospective gates must require nonzero shield
activity, exact proposal-to-action identity, exact preservation of both
reward-variance objectives, bounded service and overhead, and complete
liveness accounting. A quality acquisition should be considered only after
that systems gate passes.

## Provenance

- Frozen protocol:
  `reports/auto_research/2026-10-05-m4-shield/m4_shield_protocol.json`
- Canonical machine-readable result:
  `reports/auto_research/2026-10-05-m4-shield/m4_shield_offline_result.json`
- Protocol SHA-256:
  `1a4f9a5c57ee000cded9df3214930512d95f5b115f13e3a33070c55476c2a232`
- Failed-live archive SHA-256:
  `09b9bb555fff62b19f7e415f6e89efbf6dfc90a7394dd1933cfd064ffa24927f`

Large raw archives remain outside Git and are referenced through authenticated
SHA-256 receipts. No terminal benchmark or training-quality artifact was
opened by this analysis.

# M4-Rescue offline qualification

Status: `PASS_M4_RESCUE_OFFLINE_QUALIFICATION`

The frozen M4-Rescue policy passed every development gate on the complete
authenticated historical corpus: 20 arms, 1,280 reconstructed decisions, and
10,240 candidate-decision rows. Every source archive and raw ledger was checked
against its preserved SHA-256 receipt before replay.

## Frozen policy

M4-Rescue begins with the four-group FIFO batch. It may replace no more than
one FIFO group, must remain within 0.98--1.02 times FIFO valid actor tokens, and
must preserve or increase summed group reward variance. Among feasible batches
it maximizes imminent registered L1, then total registered L1, then reward
variance, then lower token count. It enacts a non-FIFO batch only when imminent
registered L1 strictly increases; otherwise it returns FIFO exactly.

## Canonical replay

| Quantity | Result |
|---|---:|
| Interventions | 258 / 1,280 (20.15625%) |
| Mean FIFO-group overlap | 94.96094% |
| Selected imminent L1 versus FIFO | +10.60672% |
| Selected total L1 versus FIFO | +10.30438% |
| Selected reward variance versus FIFO | +7.17819% |
| Aggregate token ratio to FIFO | 1.00009387 |
| Per-decision token-ratio range | 0.98025135--1.01988893 |

All per-decision policy constraints passed. Two complete replays produced the
same canonical selection-record SHA-256. The protocol SHA-256 is
`28cac36401f8bfa27eb6f3c9741a8c692a78c98c42fb9c6642a841a2a8bb8661`;
the machine-readable result SHA-256 is
`81a06219f12860b0c0f0e2753fec7f2e146180f3751c9cc28fccdd56d740158e`.

## Interpretation

This pass establishes deterministic feasibility and the intended proxy
behavior on historical policy-generated choice sets. Because this corpus was
used to develop the policy, it is not an independent efficacy or
generalization result. It does not estimate terminal training quality,
wall-time improvement, a counterfactual queue trajectory, or causal mediation.
It authorizes only the already specified outcome-excluded exact-actuation
qualification; a quality acquisition would require a separate preregistration
and authorization.

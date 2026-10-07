# M4-Shield live qualification result

## Conclusion

The frozen outcome-excluded live systems qualification passed:
`PASS_M4_SHIELD_QUALIFIED`.

In this 64-step qualification, M4-Shield changed five scheduler decisions
(7.8125%) relative to the reward-variance proposal. Those interventions added
685.0771417617798 aggregate imminent M4 L1 while changing both the registered
total and imminent reward-variance utilities by exactly zero. All 64 executed
actions matched the shield proposal, all searches were exact without fallback,
all service-band checks passed, and combined observer/controller duty was
0.3181748%.

This supports live actionability of the M4 signal under the frozen setting. It
does not establish a terminal training-quality improvement: the protocol
deliberately excluded quality evaluation and terminal benchmarking.

## Authenticated result

| Quantity | Value |
|---|---:|
| Learner steps / scheduler decisions | 64 / 64 |
| Shield interventions | 5 (7.8125%) |
| Aggregate imminent M4 L1 gain over reward-variance base | 685.0771417617798 |
| Total reward-variance difference | 0.0 |
| Imminent reward-variance difference | 0.0 |
| Actual action = shield proposal | 64 / 64 |
| Exact search, no fallback | 64 / 64 |
| Service band satisfied | 64 / 64 |
| Intervention overlap with base proposal | 2–3 of 4 groups |
| Intervention token ratio to FIFO | 0.9937356–1.0195228 |
| Gradient-observer duty | 0.3025863% |
| Shield decision duty | 0.0155885% |
| Combined duty | 0.3181748% |
| Analyzer runtime | 411.928961368 seconds |

The five intervention-specific imminent-L1 gains were 383.6079133,
11.2524489, 139.5057051, 119.9275904, and 30.7834841.

## Liveness

The run selected 256 groups, stale-evicted 220, and removed 16 during bounded
shutdown. It earned 57 replenishment batches, consumed 56, and ended with one
credit outstanding and zero pending candidate-excess groups. The frozen
liveness gate passed.

## Provenance

- Source commit: `cdfa1c8e56394b7d8e3e299b6524a1a286243746`
- Protocol SHA-256: `a3c8f84297c2ce60c5cd5f0f0eb0a0286555ced63bbe5b169af0e3b008aca216`
- Upstream pipeline: `71952902`
- Downstream pipeline: `71953253`
- Slurm job: `6178557`, `COMPLETED`, exit `0:0`
- Queue time: `12:04:10`
- Slurm job duration: `00:10:24`
- Workload and logs-after archives: authenticated and traversal-safe
- Six critical terminal files: byte-identical across both copies
- Result SHA-256: `df6fb97686df03b8833cd9c5358588f574058dd5f46c1773d18376ee9e35e9bf`
- OARS ledger SHA-256: `4dfd6886238db11aecc6980f24b698c9578eb4c14a99139ca5751c080adac119`

The actual Slurm command contained neither `--time` nor `--deadline`; the
deadline-suppression repair allowed allocation after a 12-hour queue wait.

## Scientific boundary

This was a prospectively frozen live systems qualification of a policy chosen
using prior authenticated data. It is strong evidence that the proposed signal
can drive safe, measurable scheduler actions, but it is not independent policy
discovery and not a quality endpoint. Any claim about final model quality still
requires a separately preregistered causal outcome study.

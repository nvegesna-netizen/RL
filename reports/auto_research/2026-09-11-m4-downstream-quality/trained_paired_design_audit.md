# Adversarial audit of the trained paired design

Status: `PASS_WITH_IMPLEMENTATION_GATES`

| Reviewer challenge | Resolution | Remaining gate |
| --- | --- | --- |
| V8 is being reused as a control. | It is excluded from the estimator and used only for endpoint feasibility and a pretreatment prompt covariate. | Analyzer must reject V8 as an acquisition run. |
| The 1,024 prompts are treated as 1,024 replicates. | The training-seed block is the causal unit; prompt-level work is sensitivity only. | Analyzer must use 15 primary degrees of freedom. |
| The treatment differs in more than delay. | Within-block configs must be equal after removing release policy, unique output paths, domains, and bookkeeping IDs. | Byte-level normalized-config comparison. |
| A mixed treatment dilutes the effect. | That is deliberate: it reproduces the accepted 1:1 M4 policy and asks whether that deployed measurement regime changes learning. | Paper must not describe it as universal d5 exposure. |
| A quality effect is automatically attributed to M4. | The claim is the total effect of release policy; M4 is a randomized-policy intermediate and exclusive mediation is not claimed. | Wording test in final analyzer/report. |
| Seeds or failed runs can be replaced after seeing results. | All 16 blocks are enumerated now; no automatic retry, replacement, or extension exists. | Outcome embargo and authority comparison. |
| Best checkpoint or validation traffic changes the system. | Only the exact terminal export at update/version 448 is scored in an independent evaluator. | Config rejects in-loop validation and checkpoint selection. |
| The practical threshold is post hoc. | The 0.02 absolute margin is frozen before trained outcomes and tied transparently to 20.5/1,024 answers and 5.55% of V8 accuracy. | Protocol hash must be embedded in every manifest. |
| Four or fewer RL seeds are too unstable. | Sixteen independent blocks are fixed and randomization inference enumerates all sign flips. | Verify unique seeds and block identities. |
| Outcome-guided stopping biases the result. | Scores remain embargoed until the fixed 16-pair completion gate; no sequential analysis exists. | Detached embargo/analysis tooling. |
| Operational noncompliance is silently excluded. | ITT remains primary; mechanism/compliance only qualifies interpretation. | Analyzer must forbid as-treated primary output. |
| Compute can expand indefinitely. | Per-run and aggregate caps are frozen, with no extension rule. | Manifest validator and separate launch authority. |

The design is scientifically ready for local implementation, not for launch.
The implementation must still produce two validated config templates, 32
resolved and hashed run identities, an outcome-embargo contract, and a detached
analyzer before any training authority is accepted.

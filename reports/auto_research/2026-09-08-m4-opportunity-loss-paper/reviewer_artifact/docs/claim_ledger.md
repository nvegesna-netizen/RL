# M4 paper claim ledger

This ledger separates validated measurement claims, registered causal results,
secondary synthesis, retrospective publication analyses, and unsupported
extensions.

| ID | Proposed claim | Evidentiary role | Decision | Canonical evidence |
| --- | --- | --- | --- | --- |
| C1 | The M4 instrument records pre-release opportunity, randomized release assignment, terminal disposition, observer duty, and the direct-chain/version-advance mechanism under a protocol-bound ledger contract. | Validated instrument | Supported | `protocols/public_protocols.json`; `data/provenance.json` |
| C2 | A five-second controlled release delay produces the registered direct-chain and version-advance mechanism in the tested asynchronous GRPO runs. | Registered mechanism endpoints | Supported in every terminal scientific cell | Canonical acquisition results; mechanism conclusion `REPLICATED` throughout the Qwen campaign and all eight Llama acquisitions |
| C3 | The initial 224-step OpenMath acquisition decisively established materiality. | Historical registered primary endpoint | Not supported | Estimate 0.2098, envelope [0.1160, 0.3015], `INCONCLUSIVE` |
| C4 | The prospectively redesigned Qwen3-0.6B/OpenMath follow-up established a material M4 effect in its tested setting. | Registered primary endpoint | Supported | Estimate 0.3054, envelope [0.2759, 0.3349], `MATERIAL` |
| C5 | Materiality transported to Qwen3-1.7B on OpenMath. | Registered primary transport endpoint | Supported | Estimate 0.3025, envelope [0.2769, 0.3281], `MATERIAL` |
| C6 | The M4 effect is material in every tested model/workload cell. | Cross-cell descriptive claim | Not supported | Qwen3-1.7B/GSM8K is `NOT_MATERIAL`, and Llama 3.2 3B/GSM8K is positive but `INCONCLUSIVE` at 0.20 |
| C7 | The magnitude of M4 opportunity loss is heterogeneous across the tested model×workload grid. | Registered interactions plus retrospective joint synthesis | Supported in the tested grid | Both registered reference interactions exclude zero; dependency-aware joint test χ²(2)=11.53, p=0.00314 |
| C8 | NuminaMath prospectively extends the direction of the OpenMath-versus-GSM8K interaction. | Registered primary interaction | Supported | Estimate -0.0726, outer envelope [-0.1305, -0.0149], `SAME_DIRECTION_GENERALIZATION` |
| C9 | OpenMath and NuminaMath are two wholly independent four-cell replications. | Cross-study independence claim | Not supported | Both interactions reuse the same Qwen3-0.6B and Qwen3-1.7B GSM8K cells; joint bootstrap correlation 0.405 |
| C10 | The result is robust to HAC versus circular-block bootstrap uncertainty. | Retrospective robustness description | Supported | Qwen conclusions and all four Llama size/workload syntheses agree under the conservative outer-envelope construction; both Qwen interaction interval families and both Llama size-contrast interval families agree |
| C11 | Terminal missingness drives the findings. | Retrospective sensitivity claim | Not supported | All 14 definitive acquisitions have terminal dispositions; lower and upper sharp endpoints coincide |
| C12 | Pre-treatment adjustment creates the positive effects. | Retrospective sensitivity claim | Not supported | Adjusted and unadjusted estimates are positive throughout; 3B combined unadjusted estimates are 0.2358 on OpenMath and 0.2635 on GSM8K |
| C13 | The opportunity-loss measurements alone prove a final reward, accuracy, convergence, or benchmark-quality effect. | Downstream outcome claim | Unsupported and prohibited | The 16-pair mixed-d5 estimate was -0.04492 [-0.13394, 0.04410]. The initial 10-pair OARS/FIFO secondary accuracy estimate was 0.0998 [0.0275, 0.1720], but the 18-block quality-primary absolute-M4/FIFO estimate was -0.0331 [-0.1162, 0.0499]. None identifies mediation through M4 alone. |
| C14 | A material M4 opportunity-loss effect appears in the tested Llama 3.2 1B setting on OpenMath and GSM8K. | Prospective replicated cross-family extension | Supported within the tested setting | Two acquisitions per workload; OpenMath 0.3991 [0.3504, 0.4467], GSM8K 0.3461 [0.2952, 0.3898]; both `MATERIAL` |
| C15 | The evidence identifies a pure architecture effect or generalizes across Llama models, all model families, RL algorithms, hardware, or deployment environments. | Broad external-validity claim | Unsupported and prohibited | Two Llama sizes were tested, but family and size were not randomized and all runs retain the same algorithm and one accelerator environment |
| C16 | The Llama 3.2 3B extension jointly confirms materiality on both workloads. | Prospective co-primary size extension | Not supported | OpenMath 0.2621 [0.2215, 0.2994] is `MATERIAL`; GSM8K 0.2135 [0.1872, 0.2398] is `INCONCLUSIVE`, so joint success is false |
| C17 | Opportunity loss is lower at Llama 3.2 3B than 1B in both tested workload configurations. | Prespecified fixed-configuration secondary contrasts | Supported | 3B-minus-1B is -0.1370 on OpenMath and -0.1326 on GSM8K; both simultaneous intervals exclude zero |
| C18 | The Llama results establish a causal or monotonic model-size scaling law. | Broad scaling claim | Unsupported and prohibited | Size was not randomized, only 1B and 3B were tested, and the cross-workload attenuation difference is inconclusive |
| C19 | A prospectively randomized end-to-end study tested the total effect of the accepted mixed-d5 policy on terminal OpenMath accuracy relative to all-immediate release. | Downstream total-effect endpoint | Supported within the tested Qwen3-0.6B/OpenMath setting | All 16 matched training-seed blocks and 32 runs authenticated; mixed-d5 minus immediate was -0.04492 with paired 95% CI [-0.13394, 0.04410], `INCONCLUSIVE` at the registered 0.02 margin |
| C20 | The downstream study precisely establishes either a meaningful terminal-accuracy decrease or practical equivalence. | Downstream decision claim | Not supported | The 95% interval crosses zero and both registered ±0.02 relevance bounds; the estimate does not resolve the direction or practically relevant magnitude of the accuracy difference. The exact sign-flip p-value is 0.29816, and all registered margin sensitivities remain `INCONCLUSIVE`. |
| C21 | Consumed-rollout staleness identifies pre-release opportunity attrition. | Measurement-identification claim | Not supported | Staleness conditions on selection and is undefined for candidates that never train. Across all six Qwen cells and eight Llama acquisitions, M4 retained 23,254 of 28,117 positive-opportunity assignments that were lost before learning and are absent from consumed-only metrics. |
| C22 | The randomized mixed-d5 regime moved the M4 mechanism during the downstream-quality study. | Prespecified secondary analysis | Supported in the tested setting | Across 16 blocks, run-wide normalized realized opportunity loss increased by 0.00739 [0.00181, 0.01296], direct-chain rate by 0.12736 [0.12213, 0.13258], and version advance by 0.20699 [0.19484, 0.21913]. |
| C23 | Block-level M4 variation explains the downstream accuracy contrasts. | Descriptive mediation diagnostic | Not supported | Correlation -0.096; descriptive slope -1.53 with 95% interval [-10.65, 7.58]. This is not a causal mediation estimate. |
| C24 | The within-acquisition M4 contrast is invariant to the prevalence of delayed groups. | Interference/generalization claim | Unsupported and prohibited | Groups share generation, ready-buffer, and learner resources. Current effects are direct assignment effects under the tested equal-mass, approximately 50% delay saturation; other saturations remain untested. |
| C25 | The primary OARS-minus-FIFO retained registered L1 opportunity estimate was 302.03. | Prospective randomized policy primary endpoint | Supported as a numerical result | Across 10 matched Llama-3.2-1B/GSM8K run pairs, the paired 95% CI was [-334.21, 938.26] and the exact sign-flip p-value was 0.3223 |
| C26 | The prespecified terminal GSM8K estimate favored OARS over FIFO in the tested 64-update Llama-3.2-1B setting. | Prospective randomized policy secondary endpoint | Supported within the tested setting | OARS minus FIFO accuracy was +0.0998 with paired 95% CI [0.0275, 0.1720]; seven pair differences were positive, two zero, and one negative. This endpoint was not the confirmatory success gate. |
| C27 | OARS met its registered policy-compliance and wall-time utility requirements in the tested setting. | Prospective randomized policy systems endpoints | Supported within the tested setting | All admitted decisions complied; geometric-mean OARS/FIFO time-to-update-64 ratio was 0.7668 with 95% CI [0.6635, 0.8861]. The valid-actor-token ratio was imprecise: 0.9800 [0.5281, 1.8188]. |
| C28 | The OARS experiments prove a general production-scheduler benefit or that retained M4 opportunity mediated a quality difference. | Policy generalization and mediation claim | Unsupported and prohibited | Both policy studies use one model, workload, 64-update horizon, and environment. The initial primary opportunity interval crossed zero; the quality-primary follow-up did not show an accuracy gain; no mediation estimand was registered. |
| C29 | A prospectively frozen three-arm study made absolute-M4-minus-FIFO terminal accuracy the sole primary scheduler endpoint. | Prospective randomized quality-primary endpoint | Supported as a completed design and numerical result | All 18 matched blocks and 54 authenticated runs entered the frozen analysis. The estimate was -0.0331 with 95% CI [-0.1162, 0.0499] and exact sign-flip p=0.4038. |
| C30 | Absolute-M4 scheduling improved terminal GSM8K accuracy relative to FIFO in the quality-primary study. | Primary scheduler-benefit claim | Not supported | The estimate was -0.0331 [-0.1162, 0.0499]; the preregistered lower-bound-above-zero criterion was not met. |
| C31 | Reward-variance scheduling increased retained registered L1 opportunity relative to FIFO in the quality-primary study. | Registered mechanism statistic | Supported in the tested setting | Reward variance minus FIFO was 311.53 [125.40, 497.66]. Absolute M4 minus FIFO was 215.51 [-7.82, 438.84]. |
| C32 | The active policies reduced time to update 64 relative to FIFO in the quality-primary study. | Registered systems statistic | Supported in the tested setting | Geometric-mean ratios were 0.8023 [0.7327, 0.8785] for reward variance/FIFO and 0.8139 [0.7435, 0.8910] for absolute M4/FIFO. |
| C33 | The initial positive OARS secondary accuracy result establishes a stable or replicated scheduler-quality gain. | Cross-study policy synthesis | Not supported | The larger quality-primary follow-up estimated reward variance minus FIFO at -0.0109 [-0.1062, 0.0844] and absolute M4 minus FIFO at -0.0331 [-0.1162, 0.0499]. The combined evidence supports actionability and heterogeneity, not a stable quality gain. |
| C34 | A prospectively frozen M4-Shield qualification demonstrated constrained live actuation of the M4 signal. | Outcome-excluded live systems qualification | Supported within the tested 64-decision setting | M4-Shield intervened on 5 of 64 decisions; enacted actions matched the shield proposal 64/64 times, exact search used no fallback 64/64 times, and every decision remained inside the registered service band. |
| C35 | M4-Shield improved imminent M4 opportunity relative to the reward-variance base proposal while preserving the registered reward-variance utilities. | Frozen qualification endpoints | Supported within the tested setting | Aggregate imminent L1 gain was +685.08. Total and imminent reward-variance utility differences were both exactly 0.0; combined gradient-observer and decision duty was 0.3182%. |
| C36 | The M4-Shield qualification establishes a terminal-quality improvement. | Downstream outcome claim | Unsupported and prohibited | Training quality was deliberately excluded from the qualification; no terminal-quality outcome was acquired or analyzed. |
| C37 | M4-Shield is a production-ready or generally superior asynchronous scheduler. | Deployment and generalization claim | Unsupported and prohibited | The qualification contains one 64-decision run and five interventions in one tested model/workload/environment. It establishes constrained live feasibility, not comparative deployment performance or generalization. |

## Canonical wording

Use:

> In the tested asynchronous GRPO settings, a controlled five-second release
> delay causally increased normalized gradient-opportunity loss. The effect was
> material in five of six definitive Qwen3×math-workload cells and exhibited
> model-by-workload heterogeneity. A prospective Llama 3.2 1B extension then
> replicated material effects twice within each of two tested workloads. A
> preregistered 3B extension remained positive, confirmed materiality on
> OpenMath, and showed lower opportunity loss than 1B in both fixed workload
> configurations. A separate prospective 16-pair run-randomized study then
> tested terminal OpenMath accuracy; its negative estimate was not precise
> enough to determine the direction or practically relevant magnitude of the
> accuracy difference. Finally, a 10-pair randomized Llama-3.2-1B/GSM8K policy
> study compared baseline-budgeted OARS with FIFO. Its registered primary
> retained-opportunity estimate was 302.03 (95% CI [-334.21, 938.26]); the
> prespecified secondary terminal-accuracy estimate favored OARS by 9.98
> percentage points (95% CI [2.75, 17.20]). A larger prospectively frozen
> three-arm follow-up then made terminal accuracy primary. Absolute M4 minus
> FIFO was -3.31 percentage points (95% CI [-11.62, 4.99]), while reward
> variance increased retained opportunity and both active policies reduced wall
> time. A separate outcome-excluded live qualification then evaluated
> M4-Shield, a constrained repair layer over reward-variance scheduling. It
> changed 5 of 64 decisions, added 685.08 imminent L1 opportunity relative to
> the base proposals, preserved both registered reward-variance utilities
> exactly, and met the service, liveness, and overhead gates. Together these
> studies show that M4 can diagnose loss and drive bounded scheduler actions,
> but they do not establish a stable terminal-quality improvement.

Do not use “confirmed final-quality decrease,” “equivalent final quality,”
“M4 mediated the OARS quality effect,”
“replicated scheduler-quality gain,” “M4-Shield improved terminal quality,”
“production-ready scheduler,” “general across LLMs,” “pure family effect,”
“general scaling law,” or language that retroactively reclassifies the initial
acquisition or calls the 3B co-primary study a joint success.

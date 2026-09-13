# M4 paper claim ledger

This ledger separates validated measurement claims, registered causal results,
secondary synthesis, retrospective publication analyses, and unsupported
extensions. “Pipeline success” is never used as a synonym for scientific
success.

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
| C13 | The opportunity-loss results prove a final reward, accuracy, convergence, or benchmark-quality effect. | Downstream outcome claim | Unsupported and prohibited | The separate run-randomized quality study was `INCONCLUSIVE`; it cannot promote the proximal M4 endpoint into a confirmed quality effect |
| C14 | A material M4 opportunity-loss effect appears in the tested Llama 3.2 1B setting on OpenMath and GSM8K. | Prospective replicated cross-family extension | Supported within the tested setting | Two acquisitions per workload; OpenMath 0.3991 [0.3504, 0.4467], GSM8K 0.3461 [0.2952, 0.3898]; both `MATERIAL` |
| C15 | The evidence identifies a pure architecture effect or generalizes across Llama models, all model families, RL algorithms, hardware, or deployment environments. | Broad external-validity claim | Unsupported and prohibited | Two Llama sizes were tested, but family and size were not randomized and all runs retain the same algorithm and one accelerator environment |
| C16 | The Llama 3.2 3B extension jointly confirms materiality on both workloads. | Prospective co-primary size extension | Not supported | OpenMath 0.2621 [0.2215, 0.2994] is `MATERIAL`; GSM8K 0.2135 [0.1872, 0.2398] is `INCONCLUSIVE`, so joint success is false |
| C17 | Opportunity loss is lower at Llama 3.2 3B than 1B in both tested workload configurations. | Prespecified fixed-configuration secondary contrasts | Supported | 3B-minus-1B is -0.1370 on OpenMath and -0.1326 on GSM8K; both simultaneous intervals exclude zero |
| C18 | The Llama results establish a causal or monotonic model-size scaling law. | Broad scaling claim | Unsupported and prohibited | Size was not randomized, only 1B and 3B were tested, and the cross-workload attenuation difference is inconclusive |
| C19 | A prospectively randomized end-to-end study tested the total effect of the accepted mixed-d5 policy on terminal OpenMath accuracy relative to all-immediate release. | Downstream total-effect endpoint | Supported within the tested Qwen3-0.6B/OpenMath setting | All 16 matched training-seed blocks and 32 runs authenticated; mixed-d5 minus immediate was -0.04492 with paired 95% CI [-0.13394, 0.04410], `INCONCLUSIVE` at the registered 0.02 margin |
| C20 | The downstream study precisely establishes either a meaningful terminal-accuracy decrease or practical equivalence. | Downstream decision claim | Not supported | The 95% interval crosses zero and both registered ±0.02 relevance bounds; the estimate does not resolve the direction or practically relevant magnitude of the accuracy difference. The exact sign-flip p-value is 0.29816, and all registered margin sensitivities remain `INCONCLUSIVE`. |

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
> accuracy difference.

Do not use “confirmed final-quality decrease,” “equivalent final quality,”
“general across LLMs,” “pure family effect,” “general scaling law,” or language that
retroactively reclassifies the initial acquisition or calls the 3B co-primary
study a joint success.

# M4 paper claim ledger

This ledger separates validated measurement claims, registered causal results,
secondary synthesis, retrospective publication analyses, and unsupported
extensions. “Pipeline success” is never used as a synonym for scientific
success.

| ID | Proposed claim | Evidentiary role | Decision | Canonical evidence |
| --- | --- | --- | --- | --- |
| C1 | The M4 instrument records pre-release opportunity, randomized release assignment, terminal disposition, observer duty, and the direct-chain/version-advance mechanism under a protocol-bound ledger contract. | Validated instrument | Supported | `protocols/public_protocols.json`; `data/provenance.json` |
| C2 | A five-second controlled release delay produces the registered direct-chain and version-advance mechanism in the tested asynchronous GRPO runs. | Registered mechanism endpoints | Supported in every terminal scientific cell | Canonical acquisition results; mechanism conclusion `REPLICATED` in the Qwen campaign and all four Llama V5 acquisitions |
| C3 | The initial 224-step OpenMath acquisition decisively established materiality. | Historical registered primary endpoint | Not supported | Estimate 0.2098, envelope [0.1160, 0.3015], `INCONCLUSIVE` |
| C4 | The prospectively redesigned Qwen3-0.6B/OpenMath follow-up established a material M4 effect in its tested setting. | Registered primary endpoint | Supported | Estimate 0.3054, envelope [0.2759, 0.3349], `MATERIAL` |
| C5 | Materiality transported to Qwen3-1.7B on OpenMath. | Registered primary transport endpoint | Supported | Estimate 0.3025, envelope [0.2769, 0.3281], `MATERIAL` |
| C6 | The M4 effect is material in every tested model/workload cell. | Cross-cell descriptive claim | Not supported | Five of six Qwen3 cells and both replicated Llama 3.2 1B workload syntheses are `MATERIAL`; Qwen3-1.7B/GSM8K is positive but `NOT_MATERIAL` at 0.20 |
| C7 | The magnitude of M4 opportunity loss is heterogeneous across the tested model×workload grid. | Registered interactions plus retrospective joint synthesis | Supported in the tested grid | Both registered reference interactions exclude zero; dependency-aware joint test χ²(2)=11.53, p=0.00314 |
| C8 | NuminaMath prospectively extends the direction of the OpenMath-versus-GSM8K interaction. | Registered primary interaction | Supported | Estimate -0.0726, outer envelope [-0.1305, -0.0149], `SAME_DIRECTION_GENERALIZATION` |
| C9 | OpenMath and NuminaMath are two wholly independent four-cell replications. | Cross-study independence claim | Not supported | Both interactions reuse the same Qwen3-0.6B and Qwen3-1.7B GSM8K cells; joint bootstrap correlation 0.405 |
| C10 | The result is robust to HAC versus circular-block bootstrap uncertainty. | Retrospective robustness description | Supported | Qwen conclusions and both Llama workload syntheses agree under the conservative outer-envelope construction; both Qwen interaction interval families exclude zero |
| C11 | Terminal missingness drives the findings. | Retrospective sensitivity claim | Not supported | All Qwen common-window and Llama V5 primary assignments have terminal dispositions; lower and upper sharp endpoints coincide |
| C12 | Pre-treatment adjustment creates the positive effects. | Retrospective sensitivity claim | Not supported | Adjusted and unadjusted point estimates are positive throughout; the Llama combined unadjusted estimates are 0.3600 on OpenMath and 0.4031 on GSM8K |
| C13 | The result proves improved final reward, accuracy, convergence, or benchmark quality. | Downstream outcome claim | Unsupported and prohibited | No downstream model-quality endpoint was measured |
| C14 | A material M4 opportunity-loss effect appears in the tested Llama 3.2 1B setting on OpenMath and GSM8K. | Prospective replicated cross-family extension | Supported within the tested setting | Two acquisitions per workload; OpenMath 0.3991 [0.3504, 0.4467], GSM8K 0.3461 [0.2952, 0.3898]; both `MATERIAL` |
| C15 | The evidence identifies a pure architecture effect or generalizes across Llama models, all model families, RL algorithms, hardware, or deployment environments. | Broad external-validity claim | Unsupported and prohibited | Only Llama 3.2 1B was added; Qwen and Llama were not randomized as a family factor, and all runs retain the same algorithm and one accelerator environment |

## Canonical wording

Use:

> In the tested asynchronous GRPO settings, a controlled five-second release
> delay causally increased normalized gradient-opportunity loss. The effect was
> material in five of six definitive Qwen3×math-workload cells and exhibited
> model-by-workload heterogeneity. A prospective Llama 3.2 1B extension then
> replicated material effects twice within each of two tested workloads.

Do not use “improved model quality,” “general across LLMs,” “pure family
effect,” or language that retroactively reclassifies the initial acquisition.

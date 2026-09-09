# M4 paper claim ledger

This ledger separates validated measurement claims, registered causal results,
secondary synthesis, retrospective publication analyses, and unsupported
extensions. “Pipeline success” is never used as a synonym for scientific
success.

| ID | Proposed claim | Evidentiary role | Decision | Canonical evidence |
| --- | --- | --- | --- | --- |
| C1 | The M4 instrument records pre-release opportunity, randomized release assignment, terminal disposition, observer duty, and the direct-chain/version-advance mechanism under a protocol-bound ledger contract. | Validated instrument | Supported | `evidence_map.json`; implementation commits `c0d12e61f`, `5940059c8`, `766351123`, `4e6f6a993`, `9cc2e9c6e` |
| C2 | A five-second controlled release delay produces the registered direct-chain and version-advance mechanism in the tested asynchronous GRPO runs. | Registered mechanism endpoints | Supported in every terminal scientific cell | Canonical acquisition results; mechanism conclusion `REPLICATED` |
| C3 | The initial 224-step OpenMath acquisition decisively established materiality. | Historical registered primary endpoint | Not supported | Estimate 0.2098, envelope [0.1160, 0.3015], `INCONCLUSIVE` |
| C4 | The prospectively redesigned Qwen3-0.6B/OpenMath follow-up established a material M4 effect in its tested setting. | Registered primary endpoint | Supported | Estimate 0.3054, envelope [0.2759, 0.3349], `MATERIAL` |
| C5 | Materiality transported to Qwen3-1.7B on OpenMath. | Registered primary transport endpoint | Supported | Estimate 0.3025, envelope [0.2769, 0.3281], `MATERIAL` |
| C6 | The M4 effect is material in every tested model/workload cell. | Cross-cell descriptive claim | Not supported | Five of six definitive cells are `MATERIAL`; Qwen3-1.7B/GSM8K is positive but `NOT_MATERIAL` at 0.20 |
| C7 | The magnitude of M4 opportunity loss is heterogeneous across the tested model×workload grid. | Registered interactions plus retrospective joint synthesis | Supported in the tested grid | Both registered reference interactions exclude zero; dependency-aware joint test χ²(2)=11.53, p=0.00314 |
| C8 | NuminaMath prospectively extends the direction of the OpenMath-versus-GSM8K interaction. | Registered primary interaction | Supported | Estimate -0.0726, outer envelope [-0.1305, -0.0149], `SAME_DIRECTION_GENERALIZATION` |
| C9 | OpenMath and NuminaMath are two wholly independent four-cell replications. | Cross-study independence claim | Not supported | Both interactions reuse the same Qwen3-0.6B and Qwen3-1.7B GSM8K cells; joint bootstrap correlation 0.405 |
| C10 | The result is robust to HAC versus circular-block bootstrap uncertainty. | Retrospective robustness description | Supported | All six cell conclusions agree under the conservative outer-envelope construction; both interaction interval families exclude zero |
| C11 | Terminal missingness drives the findings. | Retrospective sensitivity claim | Not supported | All six common-window cells have zero missing terminal dispositions; lower and upper sharp endpoints coincide |
| C12 | Pre-treatment adjustment creates the positive effects. | Retrospective sensitivity claim | Not supported | All six adjusted and unadjusted common-window point estimates are positive; adjustment changes magnitude and is largest at 0.6B/NuminaMath (-0.0538) |
| C13 | The result proves improved final reward, accuracy, convergence, or benchmark quality. | Downstream outcome claim | Unsupported and prohibited | No downstream model-quality endpoint was measured |
| C14 | The effect generalizes to other model families, RL algorithms, hardware, or deployment environments. | External-validity claim | Unsupported and prohibited | Tested scope is Qwen3-0.6B/1.7B, GRPO, two H100s, EOS, and three math workloads |

## Canonical wording

Use:

> In the tested asynchronous GRPO settings, a controlled five-second release
> delay causally increased normalized gradient-opportunity loss. The effect was
> material in five of six definitive Qwen3×math-workload cells and exhibited
> model-by-workload heterogeneity.

Do not use “improved model quality,” “general across LLMs,” “two independent
replications,” or language that retroactively reclassifies the initial
acquisition.

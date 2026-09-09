# Primary evidence table

The registered full-window cell results remain the authoritative cell-level
decisions. The harmonized versions 8–407 estimates are secondary inputs to the
six-cell synthesis and cannot replace those decisions.

| Workload | Model | Full-window assignments | Registered adjusted estimate | Registered 95% outer envelope | Registered cell conclusion | Common-window assignments | Common-window adjusted estimate |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| OpenMath | Qwen3-0.6B | 7,199 | 0.3054 | [0.2759, 0.3349] | `MATERIAL` | 7,199 | 0.3054 |
| OpenMath | Qwen3-1.7B | 8,673 | 0.3025 | [0.2769, 0.3281] | `MATERIAL` | 6,955 | 0.2982 |
| GSM8K | Qwen3-0.6B | 9,573 | 0.2323 | [0.2108, 0.2533] | `MATERIAL` | 7,710 | 0.2333 |
| GSM8K | Qwen3-1.7B | 9,429 | 0.1378 | [0.1152, 0.1599] | `NOT_MATERIAL` | 7,613 | 0.1431 |
| NuminaMath | Qwen3-0.6B | 7,229 | 0.3155 | [0.2878, 0.3429] | `MATERIAL` | 7,229 | 0.3155 |
| NuminaMath | Qwen3-1.7B | 7,050 | 0.2979 | [0.2615, 0.3340] | `MATERIAL` | 7,050 | 0.2979 |

The six definitive full-window acquisitions contain 49,153 assignments. Every
assignment in every harmonized cell has a scored terminal disposition.

## Dependency-aware interaction results

Let \(S_w = \Delta_{1.7B,w} - \Delta_{0.6B,w}\) be the model-scale contrast
within workload \(w\). The common-window estimates are:

| Workload | Model-scale contrast, \(S_w\) |
| --- | ---: |
| OpenMath | -0.0072 |
| GSM8K | -0.0902 |
| NuminaMath | -0.0176 |

The two reference interactions are \(S_{GSM8K}-S_{OpenMath}\) and
\(S_{GSM8K}-S_{NuminaMath}\). Both reuse the same GSM8K cells.

| Interaction | Estimate | Marginal HAC 95% interval | Marginal bootstrap 95% interval | Simultaneous bootstrap 95% interval |
| --- | ---: | ---: | ---: | ---: |
| GSM8K − OpenMath | -0.0830 | [-0.1361, -0.0299] | [-0.1366, -0.0289] | [-0.1438, -0.0222] |
| GSM8K − NuminaMath | -0.0726 | [-0.1291, -0.0160] | [-0.1305, -0.0149] | [-0.1378, -0.0073] |

Their estimated correlation is 0.379 by the HAC construction and 0.405 in the
joint bootstrap. A retrospective joint Wald test of equal model-scale effects
across all three workloads gives \(\chi^2(2)=11.53\), \(p=0.00314\). This
joint result is secondary and retrospective; it does not overwrite either
registered interaction analysis.

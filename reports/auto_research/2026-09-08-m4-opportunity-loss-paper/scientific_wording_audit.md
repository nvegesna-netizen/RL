# Scientific wording audit

Audit date: 2026-09-08

## Outcome

The manuscript supports a scoped causal systems-measurement claim. No sentence
requires a new same-design acquisition. The remaining release blockers are
human author decisions, exact dependency-version recovery, PDF compilation and
page fitting, and anonymous artifact access—not missing M4 cell evidence.

## Claim-by-claim review

| Manuscript component | Evidence test | Result | Required wording |
| --- | --- | --- | --- |
| Title and abstract | Endpoint is opportunity loss, not accuracy | Pass | Retain “opportunity loss”; do not substitute “quality loss” |
| Initial acquisition | Interval crosses registered 0.20 threshold | Pass | `INCONCLUSIVE`, despite mechanism success and completed acquisition |
| Redesigned follow-up | Design frozen before outcome; envelope above 0.20 | Pass | “prospectively redesigned confirmation” |
| Six-cell headline | Full-window counts sum to 49,153; five decisions are material | Pass | Specify “definitive full-window acquisitions” |
| Positive GSM8K cell | Qwen3-1.7B interval is wholly below 0.20 | Pass | “positive but not material,” never “no effect” |
| Heterogeneity | Both registered interactions point the same way | Pass | Limit to tested model/workload executions |
| Joint synthesis | Two contrasts share both GSM8K anchors | Pass | “dependency-aware synthesis,” never “two independent replications” |
| Mechanism | Direct-chain/version-advance checks replicate | Pass | Mechanism is a support condition, not the primary endpoint |
| Observer duty | Six values are 0.001798--0.003286, below 0.01 | Pass | Low measured duty in tested runs, not general portability |
| Missingness | No unscored definitive assignments | Pass | Endpoint bounds coincide; do not imply missingness methods were unnecessary prospectively |
| Pipeline status | Pipeline 66675506 failed after acquisition at analyzer invocation | Corrected | Separate pipeline state from scientific acquisition completion |
| Final quality | No accuracy/reward/convergence endpoint | Pass | Explicitly unsupported and a possible new study |
| External validity | Two small Qwen3 models, three math datasets, GRPO, EOS, two H100s | Pass | No model-family, workload-family, algorithm, or hardware generalization |

## Number reconciliation

The offline verifier authenticates the publication synthesis at SHA-256
`fd4c74c245b5294be175e2117c136e5cc7a3dcd5ce302713ba8041768c6902be`
and independently sums the canonical full-window counts to 49,153. It also
checks the 43,756 common-window assignments, zero terminal missingness, joint
correlations, six terminal artifact hashes, pipeline-failure wording, SVG
syntax, and BibTeX key closure.

The main table uses registered full-window results. The forest plot uses the
harmonized versions 8--407 common window. The captions and surrounding text now
state this distinction. Common-window estimates must not silently replace
registered cell decisions.

## Corrections made during audit

1. Added the omitted Qwen3-0.6B/GSM8K observer-duty value to the Markdown draft.
2. Labeled pipeline 66675506 failed and explained the post-run analyzer error.
3. Removed titles embedded inside plots to comply with MLSys figure guidance.
4. Replaced Mermaid-only diagrams with vector SVG sources.
5. Changed “representative” observer values to a complete six-cell list.
6. Kept raw-artifact access in future tense; no public availability is claimed.
7. Added an authenticated cadence audit while explicitly prohibiting post hoc
   dose renormalization.

## Open release blockers

- Final author roster, affiliations, acknowledgments, and conflict metadata.
- Recovery or explicit omission of exact driver, CUDA, PyTorch, vLLM,
  Megatron, CPU, and GPU-memory versions.
- Type-1-compatible outlining of figure fonts for final conference upload.
- Approval and hosting of the already scrubbed and locally clean-room-tested
  anonymous compact artifact; raw-ledger release requires separate review.
- Original operational rationale for the 0.20 threshold, if a pre-outcome
  design record contains one. Do not invent a retrospective rationale.

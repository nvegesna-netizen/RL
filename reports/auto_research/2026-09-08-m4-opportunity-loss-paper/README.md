# M4 publication package

This directory is the review-ready publication workspace for the completed M4
opportunity-loss campaign and its prospective Llama 3.2 1B and 3B extensions.
It also integrates the separately preregistered 16-pair downstream-quality
study without promoting its inconclusive result into the headline claim.

## Start here

- `manuscript.md` — complete first paper draft.
- `venue_decision.md` — MLSys 2027 Research Track decision, deadlines, and
  submission constraints.
- `claim_ledger.md` — supported, bounded, and prohibited claims.
- `primary_table.md` — registered cell results and the dependency-aware joint
  interaction table.
- `post_llama3b_reviewer_gap_audit.md` — current adversarial scientific review
  and go/no-go decision; `reviewer_gap_audit.md` preserves the earlier audit.
- `next_actions.md` — ordered submission sequence and new-compute decision.
- `reproducibility_appendix.md` — hardware/software, protocol, acquisition,
  observer-duty, and provenance details.
- `artifact_access_plan.md` — anonymized reviewer-artifact design, completed
  local clean-room test, and remaining public-release gates.
- `reviewer_artifact/` — scrubbed, standard-library-only compact reviewer
  bundle with opaque provenance commitments and a synthetic replay path.
- `build_reviewer_artifact.py` and `clean_room_test_reviewer_artifact.py` —
  deterministic archive builder and credential-stripped extraction test.
- `verified_sources.md` and `references.bib` — primary-record bibliography
  audit and BibTeX database.

## Analysis and figures

- `run_publication_synthesis.py` — authenticates all six Qwen raw cell inputs,
  reconstructs common-window inferences, and generates the files below.
- `run_llama_v5_publication_extension.py` — reconstructs all four Llama V5
  acquisitions and the two registered equal-replicate workload endpoints.
- `llama_v5_publication_extension.json` — machine-readable Llama extension and
  offline robustness record.
- `run_llama3b_publication_extension.py` and
  `llama_3b_publication_extension.json` — reconstruct the four 3B acquisitions,
  robustness views, and prespecified 3B-minus-1B contrasts.
- `six_cell_synthesis.json` — dependency-aware 2×3 synthesis.
- `robustness_results.json` and `robustness.md` — HAC/bootstrap,
  adjusted/unadjusted, missingness, threshold, and window sensitivity.
- `analyze_update_cadence.py` and `cadence_results.{json,md}` — authenticated,
  retrospective description of the five-second dose relative to update cadence.
- `cell_forest_plot.svg` and `interaction_plot.svg` — generated vector plots.
- `causal_diagram.md` and `provenance_diagram.md` — Mermaid source diagrams.

## Venue source

- `mlsys2027/paper.tex` — main-paper source using the official MLSys 2025 style
  retained for MLSys 2027.
- `mlsys2027/appendix.tex` — separately uploadable appendix source.
- `mlsys2027/README.md` — deterministic template/build instructions and
  anonymity checklist.

The synthesis is explicitly retrospective and secondary. Registered primary
cell and interaction conclusions in the canonical study records remain
unchanged.

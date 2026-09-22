# M4 publication package

This directory is the review-ready publication workspace for the completed M4
opportunity-loss campaign and its prospective Llama 3.2 1B and 3B extensions.
It also integrates the separately preregistered 16-pair downstream-quality
study and 10-pair OARS/FIFO policy study without promoting either study's
inconclusive registered primary result into the headline claim.

## Start here

- `manuscript.md` — complete first paper draft.
- `venue_decision.md` — MLSys 2027 Research Track decision, deadlines, and
  submission constraints.
- `claim_ledger.md` — supported, bounded, and prohibited claims.
- `primary_table.md` — registered cell results and the dependency-aware joint
  interaction table.
- `post_oars_reviewer_gap_audit.md` — current adversarial scientific review and
  go/no-go decision; the earlier audit files preserve prior review stages.
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
- `analyze_metric_discriminant.py` and `metric_discriminant.json` — authenticated
  assignment-level comparison of M4 with consumed-only version-age and latency
  summaries over all six Qwen cells and eight Llama acquisitions (106,653
  registered-window assignments), with a separate Qwen common-window view.
- `qwen_raw_recovery_receipt.json` — compact provenance for the six recovered
  terminal archives and all 12 hash-authenticated Qwen ledgers; large archives
  remain outside Git.
- `../2026-09-11-m4-downstream-quality/analyze_trained_paired_secondary.py` —
  reconstructs the frozen downstream mechanism, systems, and mediation-diagnostic
  endpoints from all 32 authenticated workload archives.
- `../2026-09-15-m4-oars-randomized/confirmatory_terminal_analysis_result.json`
  — frozen paired OARS/FIFO opportunity, quality, wall-time, and training-dose
  results from 20 authenticated runs.
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

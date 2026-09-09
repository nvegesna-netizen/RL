# M4 publication package

This directory is the review-ready publication workspace for the completed M4
opportunity-loss campaign. No new acquisition or qualification data enter any
file here.

## Start here

- `manuscript.md` — complete first paper draft.
- `venue_decision.md` — MLSys 2027 Research Track decision, deadlines, and
  submission constraints.
- `claim_ledger.md` — supported, bounded, and prohibited claims.
- `primary_table.md` — registered cell results and the dependency-aware joint
  interaction table.
- `reviewer_gap_audit.md` — adversarial review and go/no-go decision.
- `next_actions.md` — ordered submission sequence and new-compute decision.
- `reproducibility_appendix.md` — hardware/software, protocol, acquisition,
  observer-duty, and provenance details.
- `artifact_access_plan.md` — anonymized reviewer-artifact design and release
  gates; it is a plan, not evidence that artifacts are already public.
- `verified_sources.md` and `references.bib` — primary-record bibliography
  audit and BibTeX database.

## Analysis and figures

- `run_publication_synthesis.py` — authenticates all six raw cell inputs,
  reconstructs common-window inferences, and generates the files below.
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

# M4 publication package

This directory is the review-ready publication workspace for the completed M4
opportunity-loss campaign. No new acquisition or qualification data enter any
file here.

## Start here

- `manuscript.md` — complete first paper draft.
- `claim_ledger.md` — supported, bounded, and prohibited claims.
- `primary_table.md` — registered cell results and the dependency-aware joint
  interaction table.
- `reviewer_gap_audit.md` — adversarial review and go/no-go decision.

## Analysis and figures

- `run_publication_synthesis.py` — authenticates all six raw cell inputs,
  reconstructs common-window inferences, and generates the files below.
- `six_cell_synthesis.json` — dependency-aware 2×3 synthesis.
- `robustness_results.json` and `robustness.md` — HAC/bootstrap,
  adjusted/unadjusted, missingness, threshold, and window sensitivity.
- `cell_forest_plot.svg` and `interaction_plot.svg` — generated vector plots.
- `causal_diagram.md` and `provenance_diagram.md` — Mermaid source diagrams.

The synthesis is explicitly retrospective and secondary. Registered primary
cell and interaction conclusions in the canonical study records remain
unchanged.

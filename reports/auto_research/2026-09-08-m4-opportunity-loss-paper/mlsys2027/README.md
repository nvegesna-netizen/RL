# MLSys 2027 source

`paper.tex` is the anonymous main paper. `appendix.tex` is the separately
uploaded supplement. Both use the official MLSys 2025 style that the [MLSys
2027 call](https://mlsys.org/Conferences/2027/CallForResearchPapers) requires.

## Prepare the official template

Download
`https://media.mlsys.org/Conferences/MLSYS2025/mlsys2025style.zip`, verify the
archive came from `media.mlsys.org`, and copy `mlsys2025.sty`, `mlsys2025.bst`,
`fancyhdr.sty`, `algorithm.sty`, and `algorithmic.sty` into this directory.
Those upstream files are intentionally not vendored in this research commit.

Convert the four SVG figure sources to PDF with a vector-preserving converter:

```text
inkscape ../cell_forest_plot.svg --export-type=pdf
inkscape ../interaction_plot.svg --export-type=pdf
inkscape ../causal_diagram.svg --export-type=pdf
inkscape ../provenance_diagram.svg --export-type=pdf
```

Then run `latexmk -pdf paper.tex` and `latexmk -pdf appendix.tex`. A local TeX
toolchain and SVG-to-PDF converter were not present in the captured development
environment, so PDF compilation remains an explicit release gate rather than a
claimed check.

## Before submission

- Replace the placeholder author record in the source with the final roster;
  blind mode hides it, but the roster must still be correct.
- Keep `\usepackage{mlsys2025}` for review and use `[accepted]` only after
  acceptance.
- Confirm the main text ends by page 10 and references begin afterward.
- Run `pdffonts` and confirm embedded Type-1 fonts, including in figures.
- Build and inspect the appendix as a separate PDF.
- Run the privacy/anonymity checks in `../artifact_access_plan.md`.
- Have all human authors approve `ai_use_statement.tex` before inclusion or
  submission-form disclosure.

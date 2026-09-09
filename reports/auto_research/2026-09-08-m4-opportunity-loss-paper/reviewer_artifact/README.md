# M4 anonymous reviewer artifact

This credential-free bundle verifies the compact published M4 results and
exercises the estimand and dependency-aware interaction path on a synthetic
miniature ledger. It requires only Python 3.10+ and the standard library.

Run from the extracted directory:

```sh
python3 analysis/replay.py --verify
python3 analysis/render_figures.py --verify
python3 -m unittest discover -s tests -v
```

`data/published_results.json` contains the six-cell common-window results after
removal of private filesystem paths. `data/provenance.json` binds opaque
acquisition IDs A1--A6 to the frozen protocol, compact ledgers, and external
terminal archives by SHA-256. The large empirical ledgers are not included;
therefore this bundle verifies compact results but does not independently
re-estimate the empirical cells. The synthetic ledger contains no experimental
observations and is used only to test the analysis structure.

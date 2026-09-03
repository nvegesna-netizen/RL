# M4 opportunity-loss prospective follow-up result

The canonical study-level synthesis is in
[`../integrated_report.md`](../integrated_report.md), with machine-readable
provenance in [`../evidence_map.json`](../evidence_map.json).

Status: complete. The registered prospective control:d5 acquisition found a
`MATERIAL` opportunity-loss effect, replicated the M4 mechanism, and supported
the portability qualifier. No retry or outcome-guided extension is authorized or
needed.

## Primary result

The adjusted opportunity-loss estimate is `0.3054232108`, with HAC standard error
`0.0150455344` and registered 95% confidence envelope
`[0.2759345053, 0.3349119163]`. The entire envelope is above the materiality
threshold of `0.20`; the one-sided materiality p-value is
`0.0000499975001`. The circular-block bootstrap interval is
`[0.2774491724, 0.3344227273]`.

The acquisition completed all 448 trainer steps and scored all 7,199 primary
assignments: 3,520 control and 3,679 d5. The terminal-coverage gate passed, no
assignment was unscored, and the missingness identification interval collapses to
the adjusted point estimate.

## Supporting results

- Mechanism: `REPLICATED`. Direct-chain events occurred for 0/3,520 control
  assignments and 891/3,679 d5 assignments, a rate contrast of
  `0.2421853765`. The d5-control mean version-advance contrast was
  `0.4052731721`.
- Portability: `SUPPORTED`. Corrected observer duty was `0.0021738545`, below
  the registered `0.01` cap.
- Continuity analysis: the unadjusted estimate was `0.3335223005`, with
  confidence envelope `[0.2635293811, 0.4014294320]`, consistent with the
  adjusted primary analysis.

## Provenance and interpretation

The correct parent pipeline was
[65985843](https://gitlab-master.nvidia.com/dl/jet/ci/-/pipelines/65985843),
with downstream pipeline `65986159` and EOS compute job `423739790`. It used
source commit `47dc1a713cbf0e07813d31c4ba70a8262214e098` and protocol SHA-256
`6631a4ca8cfc6010150d1c135a1f5821c717f0dea189a3309c5f0f58f9b0709c`.
The preserved 54,950,561-byte artifact has SHA-256
`07cddc489ea20055161f9c435d44d0a52b91113fda9eac16472bdb531dc6ffb6`.

This does not retroactively change the first confirmatory acquisition: that
earlier registered result remains `INCONCLUSIVE`. It closes the local research
question through a new, independently registered prospective follow-up whose
terminal conclusion is `MATERIAL`.

# M4 OARS randomized confirmatory result

## Conclusion

The preregistered confirmatory classification is **INCONCLUSIVE**. In the ten
paired Llama-3.2-1B-Instruct GSM8K runs, the estimated OARS-minus-FIFO retained
registered L1 opportunity was positive on average, but its 95% confidence
interval crossed zero. The study therefore does not establish the registered
primary opportunity-retention effect and does not satisfy the `MATERIAL`
classification rule.

The prespecified secondary terminal-quality estimate favors OARS: terminal
GSM8K accuracy was 9.98 percentage points higher on average, with a 95%
confidence interval from 2.75 to 17.20 percentage points. This is evidence for
higher measured terminal quality in this paired setting, but it is a secondary
endpoint rather than the confirmatory success gate. OARS also satisfied the
prespecified wall-time utility criterion.

## Design and authentication

The frozen design contained ten paired OARS/FIFO comparisons and 20 total runs.
Pairs shared training seed, assignment seed, assignment domain, model,
workload, and the update-1-through-64 analysis window. All 20 compact results
were released only after the complete authentication gate passed. Each result
matched the SHA-256 recorded in both its terminal authentication receipt and
the completion gate. The underlying evaluation-data files were not opened.

## Preregistered estimates

| Estimand | Estimate | Two-sided 95% CI | Interpretation |
| --- | ---: | ---: | --- |
| Retained registered L1 opportunity, OARS minus FIFO | 302.03 | [-334.21, 938.26] | Primary superiority did not pass |
| Terminal GSM8K accuracy, OARS minus FIFO | 0.0998 | [0.0275, 0.1720] | Prespecified secondary estimate favors OARS |
| Wall-time ratio, OARS/FIFO | 0.7668 | [0.6635, 0.8861] | Utility criterion passed |
| Valid-actor-token ratio, OARS/FIFO | 0.9800 | [0.5281, 1.8188] | Training-dose ratio is imprecise |

For the primary endpoint, the sample standard deviation was 889.40 and the
inclusive exact two-sided paired sign-flip p-value was 0.3223. Eight of ten
paired primary differences were positive and two were negative, but their
dispersion was too large to resolve the registered primary claim.

For terminal GSM8K accuracy, seven paired differences were positive, two were
zero, and one was slightly negative. The corresponding differences in correct
answers out of 1,319 prompts were 299, 269, 0, 0, 243, 51, 308, 33, -3, and
116. The reported interval is the preregistered paired-t interval; terminal
quality was not a classification gate.

The wall-time ratio corresponds to a 23.3% lower geometric-mean time to update
64 for OARS in this setting. Policy compliance passed for all admitted runs.
The valid-actor-token ratio interval is wide, so the study does not precisely
resolve whether training dose was higher or lower under OARS.

## Classification logic

The registered `MATERIAL` decision required all ten valid pairs, policy
compliance, a primary 95% confidence-interval lower bound above zero, and a
wall-time log-ratio upper bound below `log(1.10)`. Pair validity, policy
compliance, and wall-time utility passed; primary superiority did not. The
primary upper bound was above zero, so the registered `NON_MATERIAL` rule also
did not apply. The resulting classification is therefore `INCONCLUSIVE`.

## Scope

The result applies to the tested model, GSM8K workload, 64-update training
window, intervention, and runtime environment. It does not establish that the
primary opportunity-retention effect is absent, and it should not be generalized
to other model families, sizes, workloads, or training horizons without new
evidence. The positive quality interval is informative for the downstream-
quality question, but it must remain labeled as a prespecified secondary result
and should not be used to rewrite the primary endpoint after observing it.

## Provenance

- Frozen run manifest: `confirmatory_run_manifest.json`
- Completion gate: `confirmatory_terminal_authentication_gate.json`
- Result extraction receipt: `confirmatory_result_extraction_receipt.json`
- Frozen analysis plan: `confirmatory_analysis_plan.json`
- Frozen analyzer: `analyze_confirmatory_pairs.py`
- Machine-readable result: `confirmatory_terminal_analysis_result.json`
- Analysis execution receipt: `confirmatory_analysis_execution_receipt.json`

The machine-readable result reproduced byte-for-byte on a second invocation.
A separate Ruby standard-library implementation independently matched all
headline estimates, intervals, ratios, the exact sign-flip p-value, and the
per-pair terminal-quality count differences.

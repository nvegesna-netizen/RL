# Post-OARS adversarial reviewer audit

Date: 2026-09-22

## Decision

The paper remains reviewable with the OARS result included as a scoped policy
proof of concept. The result strengthens the paper by showing that the M4
opportunity signal can be used in a preregistered actuation study, but it does
not convert the paper into a scheduler-success or causal-mediation paper.

## Evidence reconciliation

- The frozen runtime source is commit
  `90dbb632026591f24c966a43edd05b1e4933358b`, archive SHA-256
  `1d9e17a430fc1a184d767413de829ad9ee1375091485b5e7e0b24572f61f7690`.
- The frozen OARS analysis record has SHA-256
  `480180bee539b88c46f36274c9627ef99f8dae1185995f5ddaffbbbf248a25a2`.
- All 10 matched pairs and 20 Llama-3.2-1B/GSM8K runs authenticated before
  evaluation payloads were opened; policy compliance passed.
- The primary OARS-minus-FIFO retained-opportunity estimate is 302.0283 with
  paired 95% CI [-334.2075, 938.2641] and exact sign-flip p=0.32227.
- The prespecified secondary terminal-accuracy estimate is 0.09977 with 95% CI
  [0.02754, 0.17201].
- The OARS/FIFO time-to-update-64 ratio is 0.76678 [0.66350, 0.88613]; the
  valid-actor-token ratio is 0.98004 [0.52808, 1.81883].
- These values agree across the terminal analysis, claim ledger, primary table,
  manuscript, supplement, reviewer artifact, and automated checks.

## Strongest reviewer objections

1. **The primary interval is wide.** Correct. The paper reports the estimate,
   interval, and exact p-value directly rather than replacing them with a
   categorical label.
2. **Accuracy may change through paths other than retained opportunity.**
   Correct. OARS changes selection, ordering, data composition, and staleness.
   No mediation estimand was registered, so the paper makes no mediation claim.
3. **The horizon and environment are narrow.** Correct. OARS was tested for 64
   updates on one model/workload and one execution environment. The paper does
   not claim convergence, production-scheduler superiority, or transport.
4. **The token-dose interval is wide.** Correct. The point ratio is near one,
   but the interval [0.5281, 1.8188] is imprecise; the paper reports it rather
   than asserting equivalent training dose.
5. **The OARS and mixed-d5 studies appear inconsistent.** They intervene on
   different policies and settings. Mixed-d5 intentionally delays an equal-mass
   subset; OARS selects ready work subject to a token ceiling. Neither is a
   replication of the other.

## Package checks

- Publication integrity verifier: pass.
- Credential-free reviewer replay and unit tests: pass.
- Deterministic reviewer archive rebuild under Python 3.13: byte-identical.
- Main paper and supplement compile with the official style: pass (7 and 4
  pages, respectively).
- Main-paper operational-status language: absent; the proper name PipelineRL
  remains only as a related-work citation.
- Remaining submission gates: human author-roster approval, external anonymous
  artifact hosting/link test, and figure-font conversion required by the venue.

## Recommendation

Include OARS in the submission. Lead with the validated causal measurement and
heterogeneity results; present OARS as evidence that the signal is actionable
enough to motivate policy design. Do not add same-design runs after observing
the confirmatory result. A future study should preregister longer-horizon,
multi-workload policy evaluation and a design capable of separating retained-
opportunity mediation from selection-composition effects.

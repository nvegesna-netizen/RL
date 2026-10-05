# Post-quality-primary adversarial reviewer audit

Date: 2026-10-05

## Decision

The paper remains suitable for bounded scientific review, but the scheduler
claim must be narrower than it was after the initial OARS study. The core paper
is a causal measurement and heterogeneity paper. The scheduler experiments add
a useful negative boundary: measured opportunity is manipulable, and active
selection can reduce runtime, but proxy movement did not produce a stable
terminal-accuracy gain across the two prospective scheduler studies.

No same-design acquisition should be added after observing these results.

## Evidence authenticated and reconciled

- All 18 matched blocks and 54 runs entered the frozen three-arm analysis.
- Absolute M4 minus FIFO terminal GSM8K accuracy was -0.0331059 with 95% CI
  [-0.1161543, 0.0499425] and exact sign-flip p=0.4038086.
- Reward variance minus FIFO accuracy was -0.0109089
  [-0.1062198, 0.0844019].
- Reward variance minus FIFO retained registered L1 opportunity was 311.5318
  [125.4025, 497.6611]. Absolute M4 minus FIFO was 215.5113
  [-7.8163, 438.8389].
- Wall-time ratios were 0.8023271 [0.7327479, 0.8785132] for reward variance
  versus FIFO and 0.8139404 [0.7435410, 0.8910053] for absolute M4 versus FIFO.
- The frozen terminal analysis SHA-256 is
  `17490153ef9521a830e5538423479c6d4af6bbd03c58bcbcf3aa5663973faa52`.
  A byte-identical replay and an independent standard-library implementation
  reproduced the arm means, contrasts, intervals, exact test, and ratios.

## Strongest reviewer objections

1. **The positive 10-pair accuracy result did not replicate.** Correct. The
   manuscript now presents the original +0.0998 [0.0275, 0.1720] secondary
   estimate together with the larger follow-up estimates. It does not call the
   scheduler-quality effect stable or replicated.
2. **M4 may be a poor optimization target even if it is a valid diagnostic.**
   Supported as a live concern. Reward variance clearly moved retained L1
   opportunity without improving the primary accuracy endpoint. The paper now
   distinguishes instrument validity, proxy actionability, and downstream
   utility.
3. **Faster completion could reflect a different training dose.** Actor-token
   ratio intervals are reported and remain wide; the manuscript does not claim
   a throughput--quality frontier or equal-dose superiority.
4. **Zero-accuracy runs dominate the endpoint.** They are part of the frozen
   estimand and were retained: 9/18 FIFO, 7/18 reward-variance, and 11/18
   absolute-M4 runs. Excluding them post hoc would be invalid.
5. **The result proves no effect or equivalence.** It does not. The paper gives
   estimates and 95% intervals and states only that the preregistered
   lower-bound-above-zero criterion was not met.
6. **Operational success is being substituted for science.** It is not.
   Operational pipeline status is absent from the scientific narrative; only
   authenticated experimental endpoints and provenance commitments enter the
   claims. `PipelineRL` appears solely as the name of cited prior work.

## Package and anonymity checks

- Main paper and supplement compile to 7 and 4 pages, respectively; visual
  inspection found no clipping or overflow.
- The compact reviewer archive contains the new public-safe 18-block summary
  and provenance hashes. It passes manifest replay, figure verification,
  standard-library tests, and the forbidden-content scan in a credential-free
  four-variable environment.
- Two independent archive builds are byte-identical. Archive SHA-256:
  `ee4664af207cd4c9adcd75f5c8b8f6a9cff34f5130b469bf0d4bcfbc84b667cf`.
- No internal pipeline or job identifier is present in the anonymous bundle.

## Remaining non-scientific release gates

- Human approval of title, abstract, related-work framing, and author roster.
- External anonymous hosting/link test and data-owner approval for any raw
  ledger release.
- Replace Type 3 glyphs inherited from SVG-derived PDF figures before the final
  venue upload.

## Recommendation

Submit the completed study after those release gates. Lead with M4's causal
measurement and heterogeneous transport result. Present scheduler actuation as
evidence that the signal can change selection and runtime while remaining an
insufficient standalone objective for terminal quality. A future scheduler
paper should preregister a composite or constrained objective and test longer
horizons and additional workloads; it should not be framed as another retry of
this frozen experiment.

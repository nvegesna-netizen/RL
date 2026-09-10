# Llama M4 successor decision evidence

Status: `LOCAL_EVIDENCE_SYNTHESIS_NO_LAUNCH_AUTHORITY`.

## Decision

Do not retry or reclassify V4. If Llama external-validity evidence remains a
paper priority, preregister a separate V5 with **two independent 448-step causal
acquisition replicas per workload**. Each replicate should retain burn-in
versions 0--7, primary versions 8--407, and terminal guard versions 408--447.
All four packages must be frozen and submitted before any causal outcome is
inspected. Qualification observations remain excluded from causal estimators.

Before packaging V5, repair the post-run analyzer's dataclass access and require
an exact offline replay against both preserved V4 cells. This is a code repair,
not authority to rerun V4 or launch V5.

## Evidence chain

| Generation | Actual boundary | OpenMath | GSM8K | Scientific use |
| --- | --- | --- | --- | --- |
| Sealed synchronous predecessor | Genuine observer-duty gate | Passed all gates; duty 0.00595336; 9,450 projected assignments | Passed 7/8; duty 0.01064760 versus 0.01; 13,612.5 projected assignments | Shows the original Llama design was close, but supplies no causal estimate |
| Lifecycle V1 | Package import failure before training | No measurement | No measurement | No scientific evidence |
| Lifecycle V2 | Gitless/full-fingerprint mismatch before imports | No measurement | No measurement | Operational diagnosis only |
| Lifecycle V3 | Both cells completed 32 steps | Terminally censored version 31 broke the frozen join; diagnostic versions 8--30 averaged 20.52 groups/version | Only version 1 missed the brittle minimum-15 rule; versions 8--30 averaged 34.30 groups/version | Validated mechanism and exposed startup/terminal gate defects |
| Lifecycle V4 | Both cells completed 64 steps; offline analyzer bug after training | Passed 10/11; 776 joined groups; projected 6,466.67; lower 6,241.67 versus 6,900 | Passed 11/11; 1,473 joined groups; projected 12,275; lower 10,625 | Direct planning evidence for a separately preregistered successor |

The synchronous GSM8K duty miss was only 0.00064760 absolute, or 6.48%
relative to the ceiling. The lifecycle-derived recorder resolves that limitation:
V4 duty was 0.00083559 for OpenMath and 0.00221739 for GSM8K, respectively
91.64% and 77.83% below the 1% ceiling.

The sealed synchronous branch also contains a consistent repair sequence rather
than a single lucky near-pass. GSM8K corrected duty moved from 0.01203030 to
0.01153856 and finally 0.01064760 while the 0.01 threshold and measurement scope
remained unchanged. Corresponding OpenMath runs were already below the ceiling
(0.00650449, 0.00499401, and 0.00595336). This is strong evidence that observer
overhead—not model access, reward support, lifecycle validity, information
yield, or resource capacity—was the sole blocker in the synchronous design.

V4 also resolves the V3 boundary problems. It excludes startup versions 0--7,
uses only common evaluable versions 8--55, and reserves versions 56--64 as a
terminal guard. Every evaluable version is nonempty and both strict joins pass.
OpenMath's only miss is therefore real capacity evidence under the registered
rule, not censoring, observer overhead, bootstrap, training, or mechanism
failure. Its lower bound is 658.33 assignments short, a 9.54% shortfall.

## Throughput stability

OpenMath's six consecutive eight-version V4 block means are:

`15.875, 15.625, 16.250, 16.375, 16.625, 16.250` groups/version.

The first and last 24-version means are 15.92 and 16.42. There is no observed
late-window decline in OpenMath. Nevertheless, V4's mean of 16.17 is materially
below V3's versions-8--30 mean of 20.52 and the synchronous predecessor's
whole-run mean of 23.63. Successor planning should therefore use V4, the newest
and most conservative evidence, rather than the more favorable historical rate.

GSM8K remains well above capacity despite a lower final block: its V4 point and
lower projections are 12,275 and 10,625. That cell is not the binding constraint.

## Prospective design comparison

For a single future run, applying the frozen V4 circular-block projection to
OpenMath gives:

| Primary versions | Projected assignments | Lower projection | Margin over 6,900 |
| ---: | ---: | ---: | ---: |
| 400 | 6,466.67 | 6,241.67 | -9.54% |
| 443 | 7,161.83 | 6,912.65 | +0.18% |
| 448 | 7,242.67 | 6,990.67 | +1.31% |
| 480 | 7,760.00 | 7,490.00 | +8.55% |
| 512 | 8,277.33 | 7,989.33 | +15.79% |
| 576 | 9,312.00 | 8,988.00 | +30.26% |

The mathematical minimum is 443 primary versions, but that margin is too thin.
A 576-version primary window would be well supported and a 624-step run would
project to 1.50 OpenMath wall-hours / 3.01 GPU-hours and 1.04 GSM8K wall-hours /
2.08 GPU-hours. It is not preferred because it changes the model-version horizon
of the estimand relative to every 400-version Qwen cell.

Two independent replicas preserve the original causal window and geometry.
A 100,000-draw, two-replicate planning bootstrap using independent circular
block resamples of the V4 count sequence gives:

| Workload | Combined point projection | Planning lower 95% projection | Sequential wall-hours | Aggregate GPU-hours |
| --- | ---: | ---: | ---: | ---: |
| OpenMath | 12,933.33 | 12,625.00 | 2.23 | 4.46 |
| GSM8K | 24,550.00 | 22,283.33 | 1.57 | 3.14 |

These are planning projections, not acquired observations. They assume the V4
neutral-run rate transports to each causal acquisition replicate and use fixed
planning seeds 20261019 and 20261020. The prospective V5 analysis must stratify
randomization and circular-block resampling by replicate, report each replicate
separately, and combine them using a frozen rule. It must not treat the replicas
as one uninterrupted version series.

For the binding OpenMath cell, the two-replicate planning lower bound remains
10,100 after a 20% throughput degradation, 8,837.5 after a 30% degradation, and
7,575 after a 40% degradation. It reaches 6,900 only after a 45.35% degradation.
This sensitivity does not model between-run failures or prove power; the V5
power analysis must add a run-level variance component before launch. It does
show why two unchanged-window replicas are less fragile than the mathematical
single-run minimum.

## Why the alternatives are weaker

- **Retry V4:** prohibited by V4 and unnecessary; its preserved data already
  classify every frozen gate.
- **Relax 6,900:** a post-hoc threshold change triggered by a miss.
- **Tune concurrency or learner/generator balance:** changes the system regime
  in which M4 opportunity loss is measured.
- **Extend one run:** cheaper, but changes the causal training horizon and
  weakens direct comparability to the completed Qwen grid.
- **Another neutral qualification:** adds selection opportunities without
  resolving a remaining unknown; V4 already measures mechanism, duty, timing,
  support distribution, and shutdown behavior.
- **OpenMath-only substitution:** loses the registered within-Llama workload
  contrast and ignores the paired scientific question.

## Required V5 sequence

1. Preserve V1--V4 as closed and label V4 `NOT_QUALIFIED`, not failed causal
   evidence.
2. Repair `JoinedOpportunityAssignment` access (`row.start_version`, not mapping
   subscripting) and regression-test exact offline recovery of both V4 results.
3. Freeze a new two-replicate power/capacity analysis, including replicate-aware
   HAC/bootstrap combination and missingness rules.
4. Reserve four fresh assignment domains and seeds; retain the accepted
   lifecycle-derived instrument, treatment arms, 400-version primary window,
   thresholds, model, workloads, and per-run geometry.
5. Build and clean-room test all four acquisition packages. No additional
   neutral qualification is scientifically required.
6. Commit and push the full protocol and package provenance before requesting
   explicit authority for the four causal launches.

No V5 launch, training, acquisition, retry, or extension is authorized by this
evidence synthesis.

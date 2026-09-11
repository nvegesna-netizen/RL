# Post-Llama-3B reviewer gap audit

## Verdict

The Llama 3B extension materially strengthens the paper by adding a second size
from a second model family and by prospectively testing size transport on two
workloads. It is a bounded positive result, not a clean joint confirmation:
OpenMath is `MATERIAL`, GSM8K is positive but `INCONCLUSIVE`, and the registered
joint-success criterion is false. The paper is scientifically sufficient for
its proximal opportunity-loss claim without another M4 acquisition.

## Adversarial checks

| Challenge | Evidence-backed disposition |
| --- | --- |
| Was the 3B study a joint replication? | No. Both co-primary workloads had to be material; GSM8K crossed 0.20. The manuscript states joint failure explicitly. |
| Is the lower 3B estimate cherry-picked? | No. Both workload contrasts were prespecified, negative, and retain simultaneous intervals below zero. All four acquisitions are reported. |
| Does this prove a scaling law? | No. Size was fixed rather than randomized and only 1B and 3B were tested. Wording is restricted to the tested configurations. |
| Could missingness or interval choice explain the result? | No terminal scores are missing. HAC and circular-block bootstrap agree on endpoint classifications and negative size contrasts. |
| Does one replicate drive the 3B conclusion? | Replicate estimates vary, especially on GSM8K. Equal-replicate combination was prospective, and leave-one-out results are disclosed rather than used to select a preferred run. |
| Is the five-second treatment equivalent across sizes? | Not established. A fixed wall-clock dose may span different update cadences; this remains a limitation rather than a post-hoc renormalization target. |
| Is more M4 acquisition required for this paper? | No. Additional sizes would incrementally broaden external validity but would not resolve the larger unmeasured link to final model quality. |

## Remaining release work

The scientific integration must be followed by an updated PDF build, exact
number/provenance verification, deterministic reviewer-artifact rebuild, and a
credential-free clean-room replay. Human author roster and final submission
approval remain outside this repository workflow.

## Recommendation

Close the current M4 acquisition program and submit the bounded claim after the
release checks pass. If a new study is opened, prioritize independently
replicated final-quality outcomes over another nearby model-size point.

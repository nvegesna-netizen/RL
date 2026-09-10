# V4 prospective paired qualification design

Status: `FROZEN_LOCAL_DESIGN_NO_LAUNCH_AUTHORITY`.

## Separation and purpose

V3 remains terminally closed at commit `fa3b70883eec5a5c71b892e36216cb2ee595b4a6`.
Its qualification observations are used only to design this new successor; they
cannot qualify V4, enter a causal estimator, or retroactively change either V3
decision.

V4 asks whether the already validated lifecycle-derived M4 instrument can
support a prospectively bounded Llama 3.2 1B acquisition on OpenMathInstruct-2
and GSM8K. It does not change the model, workloads, causal estimand, randomized
release treatment, acquisition window, materiality threshold, 1% observer-duty
ceiling, or four-hour/eight-GPU-hour limits.

## Why the qualification design changes

V3 revealed two defects in its qualification gate, not in the instrument:

1. The 32-step join treated start version 31 as opportunity-required even
   though bounded shutdown can interrupt groups at that right boundary.
2. Requiring at least 15 groups in every observed version was a brittle proxy
   for the actual acquisition requirement. GSM8K missed only startup version 1;
   OpenMath's offline diagnostic misses versions 1, 17, and the censored version
   31 despite projecting more than 6,900 primary assignments.

V4 therefore tests the actual support property over a common uncontaminated
window and quantifies projection uncertainty. This is a prospective replacement
rule, not a post-hoc V3 reanalysis.

## Fixed 64-step topology

- completed trainer steps: exactly 64;
- startup burn-in start versions: 0--7;
- common evaluable start versions: 8--55, exactly 48 version clusters;
- terminal guard start versions: 56--64;
- strict opportunity join window: 8--55 only;
- terminally censored groups are allowed only outside 8--55 and only when the
  lifecycle ledger proves `bounded_shutdown`;
- one node, two H100 GPUs, neutral zero-second arm, and unchanged Llama 3.2 1B
  training geometry per cell.

The eight-version terminal guard matches the registered acquisition bootstrap
block length and is far wider than the maximum one-version training staleness.
Start version 64 is included in the guard because generation may be assigned
after the final learner transition and before bounded shutdown completes.

## Registered gates for each cell

All gates are conjunctive:

1. Exactly 64 completed learner transitions with contiguous versions 1--64.
2. Valid neutral lifecycle topology, unique group and sample identities, binary
   reward support, and complete selected-step reconstruction.
3. A strict lifecycle/opportunity join over start versions 8--55 with 100%
   opportunity coverage and no imputation.
4. Every one of the 48 evaluable start versions has at least one joined group.
5. Let `N_v` be joined groups at evaluable version `v`. Report
   `400 * mean(N_v)`. Its one-sided 95% circular-block-bootstrap lower bound,
   using blocks of eight versions and 20,000 draws, must be at least 6,900.
6. Corrected total lifecycle-recorder duty is at most 0.01.
7. The 64-step timing projection, with the unchanged 1.25 safety factor, is at
   most four wall-hours and eight GPU-hours for 448 steps.
8. Post-run derivation metadata and all compact artifact hashes are complete.

The bootstrap operates only on the 48 ordered evaluable version counts. Its
OpenMath and GSM8K seeds are fixed in the contract. Each draw concatenates six
uniformly sampled circular eight-version blocks. The lower bound is order
statistic `ceil(0.05 * 20000) - 1` after zero-based sorting. The point projection
is reported but cannot override a lower-bound miss.

## Pair and stop rules

- Both unchanged-workload qualification packages must be committed and pushed
  before submission, then submitted before either result is inspected.
- Failure of either cell closes V4 qualification without acquisition.
- Qualification data are permanently excluded from every causal estimator.
- No automatic retry, extension, threshold change, one-cell substitution, or
  acquisition is authorized by this design.
- Package construction, clean-room testing, preflight, qualification launch,
  and acquisition each require their own subsequent authority and provenance.

## Planning evidence only

Across the uncontaminated portion available from V3 (versions 8--30), GSM8K
observed 789 groups over 23 versions (mean 34.30; projected 13,722), while
OpenMath observed 472 (mean 20.52; projected 8,209). Neither had an empty or
sub-eight group version in that window. These observations motivate feasibility
only; V4 must generate new data under fresh assignment domains and seeds.

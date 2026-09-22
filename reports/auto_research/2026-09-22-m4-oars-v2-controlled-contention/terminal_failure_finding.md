# Controlled-frontier qualification terminal finding

## Result

The frozen systems qualification did not pass. Its authenticated terminal
status is `FAIL_CONTROLLED_FRONTIER_SYSTEMS_GATE`. This run did not actuate an
OARS policy, inspect training quality, or acquire a scientific outcome.

The failure exposed a localized implementation defect. OARS-v2 scored and
recorded every asynchronous `select` poll before the underlying WeightFIFO
sampler enforced its eight-ready-group watermark. It therefore emitted 31,600
observer records for 64 actual selections. The validated v1 observer already
contains the missing pre-watermark return.

## Authenticated evidence

- Upstream pipeline: `69313635`; downstream pipeline: `69314084`.
- Workload job: `451709979`; post-run copy: `451709980`.
- Both ZIPs passed integrity. The six compact workload artifacts had exact
  manifest coverage, matched their declared SHA-256 values, and were
  byte-identical across copies before the result was opened.
- Result SHA-256:
  `adb71f77cef652ac904945612720eb446a1e9cce5aea47b7668db4ae8ad00704`.
- Raw archives remain outside Git in the ignored session directory.

## Failure anatomy

Of the 31,600 records, 31,536 were polls that selected no groups. Candidate
counts across all records were 24,379 at four, 1,089 at five, 1,049 at six,
5,019 at seven, and 64 at eight. The two 5 ms budget fallbacks both occurred on
non-selection polls. Repeated polling raised measured OARS duty to 1.1546% and
combined observer duty to 1.2066%, above the frozen 1% gate.

The 64 actual-selection records were exactly the 64 eight-candidate records.
All 64 matched controlled FIFO, all 256 scorer proposals used exact search over
70 combinations, and none fell back. Their descriptive OARS duty was 0.0068%
and descriptive combined duty was 0.0588%. These subset statistics diagnose
the defect; they do not retroactively turn the failed qualification into a
pass.

## Repair

The local repair mirrors the validated v1 control flow:

1. Return without scoring or recording while ready candidates are below the
   WeightFIFO watermark.
2. Under controlled-frontier mode, expose only the common first-eight window
   to every scorer.
3. Leave natural eager mode and the acting WeightFIFO selection unchanged.

Regression tests cover both the below-watermark no-observation rule and exact
first-eight windowing when excess ready candidates exist. Ruff, formatting,
compilation, diff checks, and all five dependency-light gate tests pass. The
focused repository unit file cannot collect locally because the workstation
environment does not contain `ray`; a dependency-complete requalification
package must run it before any EOS workload.

## Decision

The one authorized submission attempt is consumed. A replacement systems
qualification is scientifically justified because it tests a repaired
implementation defect rather than repeating an unchanged experiment, but it
requires a new explicit launch authorization. No three-arm training-quality
study should begin until that replacement passes the original frozen systems
gate.

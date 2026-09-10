# Lifecycle-derived paired neutral qualification plan

Status: `TERMINAL_FAILED_PACKAGE_BOOTSTRAP_NO_TRAINING_NO_ACQUISITION`.

Terminal evidence is recorded in `qualification_terminal_result.json` and
`terminal_close.md`. Both paired cells stopped before training at the same
package bootstrap import check. No registered qualification measurement was
produced. Per the frozen rule below, the successor is closed without retry or
acquisition.

Exactly one 32-step neutral qualification is registered for each of OpenMath
and GSM8K. Both packages must be frozen and submitted before either result is
inspected. Each uses the same exact successor source, model, topology, lifecycle
instrument, and workload geometry as its later acquisition except for the
32-step bound and the single `neutral=0s` arm.

Qualification observations estimate no treatment contrast and are permanently
barred from every causal estimator. Both cells must pass all gates:

- exactly 32 completed trainer steps and learner-version transitions;
- complete, unique lifecycle topology and 100% opportunity reconstruction;
- exact arm-blind source ordering before neutral release start;
- both binary reward values observed;
- at least 15 groups per observed start version;
- at least 6,900 projected primary-window assignments using the frozen 1.25
  safety factor;
- corrected total lifecycle-recorder duty at or below 0.01;
- projected 448-step wall time at most four hours and two-GPU use at most eight
  GPU-hours;
- successful post-run reconstruction and strict ledger join.

Either preflight failure or either qualification miss closes the successor.
There is no retry, repair cycle, threshold relaxation, extension, or one-cell
substitution.

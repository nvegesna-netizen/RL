# V3 paired neutral qualification

Status: `AUTHORIZED_NOT_SUBMITTED`.

The successful v3 bootstrap preflight permits one prospective execution of the
original paired neutral qualification with the repaired package transport. The
scientific source, model, workloads, configuration files, assignment domains,
seeds, 32-step bound, estimand exclusion, gates, and thresholds are unchanged.
The only operational delta is the v3 bootstrap already validated on EOS:

- authenticate the gitless source using exact `pyproject.toml` and `uv.lock`;
- authenticate the container submodule pins separately;
- install pinned Megatron-LM `6513e3e23d6b5eda6a1c934990b15e804237732b`
  at the path expected by `nemo_rl`;
- import `nemo_rl` immediately before `megatron`.

Exactly two cells are authorized: OpenMath and GSM8K. Each uses one node, two
GPUs, the neutral zero-second arm only, and exactly 32 completed trainer steps.
Both must be submitted with `runllm.py --no_wait` before either result is
inspected.

Each cell must independently pass the original gates: complete learner-version
topology, unique lifecycle groups, binary reward support, at least 15 groups per
observed start version, at least 6,900 projected primary-window assignments,
strict opportunity-ledger join, corrected lifecycle-recorder duty at or below
0.01, and projected 448-step duration at or below four hours and eight GPU-hours.

Qualification observations are permanently excluded from causal estimators.
Failure of either cell closes this qualification attempt. No retry, repair,
extension, one-cell substitution, scientific acquisition, or threshold change
is authorized by this plan.

# Turn-Level Credit Assignment for NeMo RL

This research project tests whether native environment rewards can provide
token-aligned credit to individual assistant turns while preserving NeMo-RL's
existing trajectory-level GRPO objective.

Status: implementation and plumbing validation. No scientific training result
is claimed yet.

## Why this project exists

NeMo-RL's environment interface already returns a reward for each interaction
turn. The standard native rollout paths sum those values into
`total_reward`, and GRPO broadcasts one trajectory advantage over every
generated token. That loses information an environment has already supplied.

Recent work reports gains from finer credit in long-horizon agents:

- [Multi-Turn GRPO](https://arxiv.org/abs/2505.11821) combines outcome and
  turn-level rewards.
- [GiGPO](https://arxiv.org/abs/2505.10978) compares actions at repeated anchor
  states.
- [HCAPO](https://arxiv.org/abs/2603.08754) uses a hindsight critic.
- [TRACE](https://arxiv.org/abs/2607.13988) derives temporal-difference credit
  from frozen-reference state values.

This first slice implements the neutral substrate needed to test native
environment rewards. It does not claim to implement TRACE.

## Supported scope

The entrypoint fails at startup unless all of these are true:

- legacy synchronous GRPO;
- native NeMo-RL environments;
- `data_plane.enabled: false`;
- synchronous generation rollouts;
- `grpo.adv_estimator.name: grpo`.

Async GRPO, TransferQueue, SingleController, NeMo Gym, GDPO, PPO,
Reinforce++, and OPD are intentionally unsupported until the research path
produces evidence for promotion.

## Method

The research entrypoint installs scoped hooks for the duration of training:

1. After each environment call, record the raw scalar reward, component
   values, and terminal flag on the latest policy-generated assistant message.
2. Before reward scaling or shaping, convert those temporary annotations into
   compact tensors:

   - `turn_rewards [B, T]`
   - `turn_credit_rewards [B, T]`
   - `turn_mask [B, T]`
   - `turn_trainable_mask [B, T]`
   - `assistant_turn_spans [B, T, 2]`
   - `turn_terminateds [B, T]`

3. Require the raw turn rewards to sum to the raw trajectory reward.
4. Remove temporary message annotations before normal message flattening.
5. Call NeMo-RL's existing `GRPOAdvantageEstimator` for the macro advantage.
6. Compute either immediate native reward or discounted return-to-go for each
   turn and scatter it only over that turn's generated assistant tokens.

The composed advantage is

```text
A(token) = macro_weight * A_GRPO(token)
         + turn_weight * native_credit(turn(token)).
```

Prompt tokens, tool observations, padding, and assistant messages supplied in
the input history never receive auxiliary credit.

With `turn_credit.enabled: false`, the core trainer runs without hooks. With
`turn_weight: 0` and `macro_weight: 1`, the estimator returns the base GRPO
advantage object directly for a strict plumbing-equivalence check.

## Configuration

The research-owned configuration is top-level so it does not change the core
GRPO schema:

```yaml
turn_credit:
  enabled: true
  source: environment
  environment_component: null
  macro_environment_component: null
  evaluation_environment_component: null
  environment_mode: immediate  # or return_to_go
  discount: 1.0
  macro_weight: 1.0
  turn_weight: 0.2
  raw_reward_atol: 1.0e-6
```

Defaults live in the local Pydantic `TurnCreditConfig`; unknown fields and
invalid numeric ranges fail during startup. For environments with named reward
components, `environment_component` selects the per-turn auxiliary source,
`macro_environment_component` selects the training trajectory objective, and
`evaluation_environment_component` selects the held-out metric. Missing or
malformed configured components fail instead of falling back to a scalar sum.

## Run the smoke recipe

From this directory:

```bash
uv run run_grpo_turn_credit.py \
  --config configs/grpo_math_0.5b_turn_credit.yaml
```

The included math recipe is a one-turn end-to-end plumbing test. It cannot
establish the value of turn-level credit because one-turn and trajectory-level
credit are equivalent.

To verify exact macro-only behavior while retaining trace plumbing:

```bash
uv run run_grpo_turn_credit.py \
  --config configs/grpo_math_0.5b_turn_credit.yaml \
  turn_credit.turn_weight=0
```

## Tests

CPU unit tests cover reward capture, component aggregation, uneven horizons,
assistant-history exclusion, empty turns, immediate and return-to-go credit,
span scattering, raw-sum validation, unsupported-path failures, and zero-weight
equivalence. They also cover malformed tensor contracts, fractional sample
multipliers, and turn-field padding/filtering through `BatchedDataDict`:

```bash
uv run --group test pytest tests/unit
```

The required one-step GPU functional test is:

```bash
uv run bash tests/functional/run_grpo_turn_credit.sh
```

The longer 1-GPU suite is:

```bash
bash \
  tests/test_suites/llm/grpo-qwen2.5-0.5b-instruct-1n1g-dtensor2tp1-turn-credit.sh
```

## Run the multi-turn pilot

The sliding-puzzle pilot uses NeMo-RL's native multi-turn environment and
reports puzzle success, turns per sample, truncation, and turn-credit reward
distributions:

```bash
uv run run_grpo_turn_credit_sliding_puzzle.py \
  --config configs/grpo_sliding_puzzle_turn_credit.yaml
```

The checked-in configuration is the macro-only control (`turn_weight: 0`). A
matched treatment changes only `turn_credit.turn_weight`, for example:

```bash
uv run run_grpo_turn_credit_sliding_puzzle.py \
  --config configs/grpo_sliding_puzzle_turn_credit.yaml \
  turn_credit.turn_weight=0.2
```

This ten-step pilot is a feasibility and signal-quality gate, not evidence of
an accuracy improvement. A performance claim requires longer matched runs and
at least three seeds.

## Run the dense-signal control

The follow-on dense puzzle is a research-only environment that keeps terminal
success as an explicit reward component and adds normalized Manhattan-potential
progress. Progress telescopes to the endpoint potential change, so repeated or
inverse moves cannot inflate cumulative progress. Its calibration cohort uses
displayed randomized goal layouts a few moves from each start, providing many
distinct easy tasks instead of repeatedly sampling a tiny fixed-goal state
space.

The checked-in configuration trains on summed potential progress, evaluates on
terminal puzzle success, and uses potential progress as the candidate localized
signal. Local credit is disabled, making it the A1 trajectory-control arm:

```bash
uv run run_grpo_turn_credit_dense_puzzle.py \
  --config configs/grpo_dense_sliding_puzzle_trajectory.yaml
```

Before changing `turn_weight`, calibrate frozen-policy rollouts and verify that
at least 80% of samples have two trainable turns, at least 30% have a nonzero
intermediate progress delta, positive/zero/negative deltas all occur, cumulative
progress telescopes, and terminal success is between 2% and 80%, inclusive. The
matched localized treatment changes only `turn_weight`; its environment,
prompts, seeds, model, training objective, and evaluation objective remain
identical.

The calibration config evaluates 256 samples without taking an optimizer step:
the dataset builder rejects duplicate boards and constructs validation before
training so the held-out validation population stays unique, disjoint, and
fixed when the training budget changes.

```bash
uv run run_grpo_turn_credit_dense_puzzle.py \
  --config configs/grpo_dense_sliding_puzzle_calibration.yaml
```

Run the zero-weight control and the proposed nonzero-weight treatment with the
same seed. Then evaluate the two logs and their `val_data_step0.jsonl` files:

```bash
uv run evaluate_dense_calibration.py \
  --control-log control.log \
  --treatment-log treatment.log \
  --control-rollouts control/val_data_step0.jsonl \
  --treatment-rollouts treatment/val_data_step0.jsonl
```

The command exits nonzero unless both arms meet the declared multi-turn,
intermediate-signal, reward-sign, and success-rate gates and their pre-update
rollout artifacts match exactly.

## Evidence required before claiming an improvement

The next scientific experiment needs a genuinely long-horizon environment with
dense native rewards or an estimated-credit backend. The included math smoke
test is not evidence of quality improvement.

The planned comparison is:

1. standard outcome-only GRPO;
2. turn trace enabled with `turn_weight: 0`;
3. immediate native turn credit;
4. discounted return-to-go credit;
5. later, TRACE estimated credit.

Use the same prompts, seeds, batch sizes, model, and compute budget. Report at
least three seeds or confidence intervals for task success, plus average turns,
truncation rate, credit distribution, and end-to-end step time. Do not select a
default credit weight or clipping policy without those measurements.

## Known limitations

- The runtime uses scoped replacement of four driver-side module functions.
  This keeps the experiment self-contained but is not the intended core API.
- Named reward components can be selected independently for training,
  evaluation, and auxiliary credit. The unmodified scalar sum is retained for
  the raw rollout-integrity check.
- Native NeMo Gym process rewards are blocked until a versioned aligned result
  contract is available; see
  [NVIDIA-NeMo/Gym#1298](https://github.com/NVIDIA-NeMo/Gym/issues/1298).
- Raw environment-reward and configured-credit distributions are emitted with
  rollout metrics. A promoted core version should additionally expose
  post-filter auxiliary-advantage metrics through a shared estimator contract.
- TRACE needs separate prefix-plus-gold-answer scoring and is a later change.

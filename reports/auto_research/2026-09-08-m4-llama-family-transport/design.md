# Llama 3.2 1B M4 family-transport design

Status: `FROZEN_LOCAL_PROTOCOL_PENDING_NO_TRAINING_PREFLIGHT`.

## Scientific question

Does the accepted M4 controlled-release instrument transport from Qwen3 to a
comparably sized dense Llama model, and does the previously observed OpenMath
versus GSM8K attenuation retain its direction?

This is a new hypothesis test. It does not modify any historical Qwen cell and
does not identify a pure causal effect of model family: tokenizer, chat
template, exact parameter count, and pre/post-training history also change.

## Prospective cells

- `meta-llama/Llama-3.2-1B-Instruct` / OpenMathInstruct-2;
- `meta-llama/Llama-3.2-1B-Instruct` / GSM8K.

Both cells are registered before either acquisition. Both acquisition packages
must be frozen and submitted before either causal outcome is inspected. A
failure in one cell does not permit inspection of the other, outcome-guided
retry, rerandomization, extension, or substitution.

## Fixed instrument and geometry

- randomized release arms: balanced `control=0s` and `d5=5s`;
- burn-in start versions: 0--7;
- primary start versions: 8--407;
- terminal guard start versions: 408--447;
- 448 trainer steps, four prompts per step, eight generations per prompt;
- windowed FIFO sampler, maximum staleness one, 16 in-flight prompts;
- one generation H100 plus one trainer H100 per cell;
- four wall-hours and eight GPU-hours maximum per cell;
- no checkpointing, automatic retry, or automatic extension.

GSM8K permits two epochs solely to reach the same fixed version window; its
epoch-specific group instance is the assignment unit. OpenMath permits one
epoch. The common 400-version window is the confirmatory basis for both cells.

## Registered conclusions

The accepted adjusted cell estimand remains

`(mean_d5(Q*D)-mean_control(Q*D))/pooled_pre_delay_mean_Q`.

Each cell uses eight-fold generalized-regression adjustment with `Q` and
`1[Q=0]`, four-lag version HAC, and 20,000 eight-version circular block
bootstrap draws. The confidence interval is the outer HAC/bootstrap envelope.

Two conclusions are reported separately:

1. **OpenMath family transport.** `MATERIAL` only when the adjusted OpenMath
   lower envelope endpoint exceeds the registered 0.20 threshold and the
   materiality test rejects at 0.05. `NOT_MATERIAL` requires its upper endpoint
   to be at most 0.20; otherwise it is `INCONCLUSIVE`.
2. **Workload-pattern transport.** For
   `Llama(OpenMath)-Llama(GSM8K)`, `SAME_DIRECTION` requires the entire 95%
   outer envelope to exceed zero, `OPPOSITE_DIRECTION` requires it to be below
   zero, and otherwise the result is `INCONCLUSIVE`.

The GSM8K direction is also frozen as supporting evidence: `POSITIVE` requires
its entire 95% envelope above zero, `NEGATIVE` requires the envelope below
zero, and otherwise it is `INCONCLUSIVE`.

The GSM8K cell's own materiality decision is supporting evidence. A positive
but sub-0.20 GSM8K result is compatible with workload-pattern transport and is
not instrument failure. Mechanism replication and observer duty are mandatory
support checks but cannot override either causal conclusion.

## Outcome-blind planning

Llama 3.2 1B was chosen because its 1.23B parameters bridge the tested Qwen3
sizes and the repository has a native Llama Megatron GRPO recipe. OpenMath and
GSM8K were selected as the most discriminating workload pair in the completed
Qwen grid. The planning alternative interpolates the two Qwen common-window
effects at 1.23B and inflates the largest scale-matched HAC standard errors by
25%. Exact calculations are frozen in `power_capacity_plan.json`.

## Gate sequence

1. Commit and push this protocol, configs, generic validators, and source lock.
2. Run exactly one shared no-training EOS preflight using
   `runllm.py --no_wait`; it may verify access, imports, conversion, tokenizer,
   datasets, config resolution, tests, packaging, and hashes, but may not train.
3. If green, separately freeze and authorize two neutral 32-step
   qualifications. Their observations can never enter a causal estimator.
4. Require both qualifications to pass model/topology, binary reward support,
   information, observer-duty, assignment-capacity, and four-hour projections.
5. Freeze both 448-step capture-only acquisition packages and submit both
   before inspecting either outcome.
6. Authenticate both terminal artifacts, then run the registered joint analysis
   exactly once and preserve compact provenance by SHA-256.

This local freeze authorizes no EOS submission by itself.

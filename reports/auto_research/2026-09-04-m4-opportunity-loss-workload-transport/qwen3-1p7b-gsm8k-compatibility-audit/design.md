# M4 Qwen3-1.7B GSM8K workload-transport compatibility audit

Status: `CONDITIONAL_GO_TO_LOCAL_PROTOCOL`. This audit authorizes only local
design and preflight implementation. It does not authorize training, GPU use,
EOS submission, neutral qualification, or scientific acquisition.

## Question

Can the accepted M4 opportunity-loss instrument test the same fixed five-second
release-delay claim on GSM8K while keeping Qwen3-1.7B, GRPO, EOS, the execution
topology, prompt contract, controlled-release mechanism, opportunity replay,
primary estimator, materiality rule, and observer-duty ceiling unchanged?

This is a one-axis workload-transport test. The intended scientific change is
the training dataset from OpenMathInstruct-2 to the `main`/`train` split of
`openai/gsm8k`.

## Compatibility evidence

The repository registers `gsm8k` as a response dataset. Its adapter converts
each example into the same two-message structure consumed by
`math_hf_data_processor`, extracts the scalar answer after `####`, and labels the
task `gsm8k`. The common math processor passes that answer as `ground_truth` to
the same `math` environment used by the completed Qwen3-1.7B acquisition.

The existing `hf_math_verify` path returns a scalar correctness reward and its
tests cover exact `0.0` and `1.0` outcomes. This preserves the accepted M4
opportunity replay's binary-reward contract, GRPO advantage estimator,
normalization, loss replay, lifecycle semantics, and pre-delay observation.
None of the controlled-release or detached-analysis code branches on dataset
identity.

The repository's `grpo_smoke.yaml` demonstrates that GRPO can select `gsm8k`,
but it is only plumbing evidence: it changes model and batch geometry, runs ten
steps, and has no M4 gates. It must not be treated as a baseline, qualification,
power result, or scientific acquisition.

## Required isolation

The workload overlay must inherit the completed Qwen3-1.7B M4 configuration and
change only the dataset selection to an explicit GSM8K `main`/`train` source.
In particular:

- keep `Qwen/Qwen3-1.7B`, GRPO, four prompts per step, eight generations per
  prompt, global batch 32, sequence limit 2,048, and generation limit 1,792;
- keep the existing `cot.txt` prompt, `math_hf_data_processor`, `math`
  environment, and `hf_math_verify`; do not substitute the separate
  `gsm8k.txt` prompt in this test;
- keep the two-GPU non-colocated single-controller topology and the windowed
  FIFO/staleness contract;
- keep the control:d5 allocation, exact five-second treatment, adjusted primary
  estimator, unadjusted supporting estimator, `Delta_L=0.2`, mechanism checks,
  missingness rules, and 1% observer-duty ceiling;
- create a new protocol identity, assignment domain and seed, output paths, and
  artifact names so no OpenMath assignment or outcome can enter this estimator.

The administrative identity changes are necessary to prevent cross-study
contamination and are not additional scientific treatment axes.

## Qualification requirement

The completed Qwen3-1.7B/OpenMath acquisition proves model and topology fit and
completed 558 steps in 8,709.859687 seconds, below EOS `batch`'s four-hour
maximum. It does not establish GSM8K assignment throughput, reward variability,
prompt-length behavior, or terminal completeness.

Therefore a short neutral qualification remains required before choosing the
confirmatory window. It must disable controlled release and causal inference and
may report only:

1. exact config resolution and model/topology fit;
2. step time and projected four-hour feasibility;
3. assignments per observed and interior trainer version;
4. reward support and non-degenerate group frequency;
5. terminal completeness and corrected observer duty.

Qualification observations cannot enter the confirmatory estimator. They may
set a fixed step window only through a prospectively declared assignment target
and guard rule.

## Go/no-go rules

Proceed to a frozen local protocol and no-training preflight package only if:

- GSM8K resolves explicitly to `openai/gsm8k`, subset `main`, split `train`;
- processed examples provide non-null scalar-answer ground truth to the unchanged
  math verifier;
- rewards remain exactly in `{0, 1}` for opportunity replay;
- all accepted M4 lifecycle, replay, inference, mechanism, and duty tests pass;
- the config diff contains no unplanned scientific changes; and
- training, acquisition, retry, and extension authority remain false.

After a green no-training preflight, proceed to neutral qualification only with
separate explicit authority. Proceed to confirmatory acquisition only if a fixed
window can meet the accepted minimum information target within a four-hour EOS
limit. If it cannot, stop this axis or prospectively redesign it; do not lower
the evidence standard or inspect treatment outcomes.

## Conclusion

Static compatibility is sufficient for `CONDITIONAL_GO_TO_LOCAL_PROTOCOL`.
GSM8K can preserve the accepted instrument's mathematical and lifecycle
contracts, and the proposed comparison isolates workload distribution more
cleanly than an immediate cross-family model test. Runtime feasibility and
information yield remain unproven and must be established by the gated sequence.

Local Python, JSON, and YAML syntax checks pass. The targeted GSM8K unit-test
selection cannot collect under the local worktree interpreter because Ray is not
installed. This is a local dependency limitation, not a test failure or runtime
qualification; the complete selected suite remains mandatory in the pinned
containerized no-training preflight.

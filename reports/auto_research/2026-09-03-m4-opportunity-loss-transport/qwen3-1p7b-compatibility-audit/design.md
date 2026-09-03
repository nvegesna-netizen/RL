# M4 Qwen3-1.7B transportability compatibility audit

Status: `CONDITIONAL_GO` for local preflight implementation. This audit grants no
GPU qualification or scientific-acquisition authority.

## Question

Can the accepted M4 opportunity-loss instrument test the same fixed five-second
release-delay claim on Qwen3-1.7B while keeping OpenMathInstruct-2, GRPO, the
control:d5 comparison, and the registered estimator unchanged?

This is a one-axis transportability test. It changes model scale within the Qwen3
family and deliberately does not change the workload at the same time.

## Compatibility result

The lifecycle, controlled-release, opportunity-replay, observer-duty, mechanism,
and detached-inference implementations contain no model-name or parameter-count
branches. Their required semantics depend on native GRPO advantages, the clipped
policy-gradient loss contract, complete lifecycle events, common pre-delay
observation, and a bounded single-controller run. Qwen3-1.7B can satisfy those
interfaces.

The completed-study guards must not be weakened or silently reused:

- `tools/opportunity_loss_followup_preflight.py` correctly requires
  `Qwen/Qwen3-0.6B`, the completed assignment identity, exact output paths, and a
  two-GPU topology.
- `tools/opportunity_loss_followup_pipeline.py` correctly accepts only protocol
  identity `m4-opportunity-loss-adjusted-followup-v1`.
- The repository contains supported Qwen3-1.7B recipes, but no already validated
  two-GPU, non-colocated, single-controller Qwen3-1.7B M4 recipe.

Therefore the accepted instrument can be carried forward without changing its
scientific semantics, but the new study needs a new protocol identity, config,
randomization domain and seed, output paths, preflight lock, and detached result
contract.

## Resource evidence

The completed Qwen3-0.6B follow-up used two GPUs for 10,264.735831 seconds:
2.8513 hours of wall time and 5.7026 GPU-hours. It completed 448 trainer steps at
22.9124 seconds per step and produced 7,199 assignments across 400 primary start
versions, or 17.9975 assignments per primary version.

At that unchanged assignment rate, the original 7,395-assignment planning
requirement would need 411 primary versions; a preferred 7,500-assignment target
would need 417. This conversion is planning evidence only because a larger model
can change the trainer-to-generation throughput ratio.

The parameter-count ratio is 1.7/0.6 = 2.833. Linear scaling would imply roughly
8.08 hours on the same two-GPU topology, or 16.16 GPU-hours before contingency.
That estimate is not a valid acquisition budget: training, generation, device
memory, and their overlap do not scale linearly. It does establish that copying
the old four-hour wall-clock cap would be unsafe.

## Required neutral qualification

Static inspection cannot select the GPU topology or trainer-step window. After a
local no-training preflight package is implemented, the smallest informative
runtime action would be a 32-step neutral qualification with controlled release
disabled.

Its allowed outputs are configuration resolution, device-memory fit, observer
duty, assignment and step throughput, and terminal completeness. It must not
produce a control-versus-d5 contrast, materiality conclusion, or outcome-guided
sample-size choice. Qualification data cannot enter the confirmatory estimator.

The qualification should answer:

1. Does one generation GPU plus one trainer GPU fit Qwen3-1.7B at the frozen
   sequence and batch geometry?
2. If not, what is the smallest supported non-colocated topology?
3. How many primary assignments per start version should be expected?
4. What fixed trainer-step window reaches at least 7,395 and preferably 7,500
   assignments with a prespecified guard window?
5. Is the projected acquisition bounded by a proportionate GPU-hour and
   wall-clock cap?

## Go/no-go rule

Proceed to a frozen neutral qualification package only if local config resolution,
instrument tests, analyzer tests, hash binding, and no-training assertions pass.
Proceed from qualification to a confirmatory design only if the accepted M4
semantics remain unchanged and the information target fits a prespecified budget.

If the model requires semantic changes to the instrument, or a proportionate
budget cannot reach the information target, stop this axis. A same-model workload
transportability study would then be preferable to lowering the evidence standard.

## Local preflight implementation

The versioned 32-step, single-neutral-arm Qwen3-1.7B overlay and a new fail-closed
no-training lock builder are implemented. The lock validates the model, dataset,
estimator prerequisites, output paths, smallest candidate topology, compatibility
audit, and absence of qualification or acquisition authority.

Python compilation, JSON and YAML syntax, Ruff checks, and Ruff formatting pass.
A lightweight isolated test run reached the repository import graph but correctly
could not substitute for the full NeMo runtime: configuration imports expand into
Ray, PyTorch, TensorDict, Transformers, datasets, and the repository workspace.
Exact config resolution and the focused unit test therefore remain requirements
of the pinned containerized no-training preflight.

The next step is to review and commit this local package, then construct a pinned
no-training manifest. Do not launch the neutral qualification until that preflight
is terminal green and its artifact proves that no training or acquisition began.

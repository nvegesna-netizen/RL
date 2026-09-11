# Credential-free no-training preflight plan

Status: `LOCAL_PLAN_ONLY_NOT_PACKAGED_NOT_LAUNCHED`

This preflight is a separate compute boundary. It loads and exports the anchor
model but performs no optimizer step, qualification, pilot, acquisition, retry,
or extension. A successful pipeline is only an implementation result.

## Purpose

The preflight must establish the entire artifact path that the scientific runs
will rely on:

> frozen config → SingleController-compatible policy construction → terminal-
> format weight save → finalized Megatron checkpoint → Megatron-to-HF conversion
> → vLLM reload → fixed held-out scoring → authenticated compact result

The accepted M4 recipe uses the Megatron training backend. The repository's
standalone evaluator uses vLLM and therefore expects a Hugging Face-readable
model. Merely proving that Megatron wrote files is insufficient.

## Frozen no-training stages

1. Materialize a clean source tree from a source archive bound to one commit.
2. Verify the source archive, image, converter, config, dataset-selection code,
   and authority record by SHA-256 before importing NeMo RL.
3. Load `Qwen/Qwen3-0.6B` with the exact training backend and tokenizer settings.
4. Save the unmodified initial policy through the same policy
   `save_checkpoint`/`finalize_async_save` methods used by terminal export. Do
   not construct an optimizer step or call a training method.
5. Verify one complete Megatron iteration and its `run_config.yaml`, then convert
   it with `examples/converters/convert_megatron_to_hf.py`.
6. Reload the converted directory through the same vLLM generation backend and
   decoding settings planned for scientific evaluation.
7. Construct the OpenMathInstruct-2 split with split seed `20260911`; select the
   candidate 1,024 evaluation rows by an outcome-independent deterministic rule;
   write ordered row fingerprints and a manifest hash.
8. Score the unmodified base model on that set. This score is used only for
   floor/ceiling, runtime, and score-variance feasibility gates. It is not a
   delay-regime outcome and cannot enter a treatment-effect estimate.
9. Write a compact result containing stage exit codes, zero optimizer steps,
   zero learner-version advances, model/config/prompt/output hashes, wall time,
   and GPU hours. Archive large weights and per-prompt outputs outside Git.

## Fail-closed gates

The preflight passes only if all of the following hold:

- no credential value is embedded in the workload or recorded artifacts;
- trainer-step and learner-version counters are both zero;
- no controlled-delay assignment or M4 causal estimate is produced;
- checkpoint finalization completes before conversion begins;
- exactly one checkpoint iteration is resolved, with no ambiguous fallback;
- conversion and vLLM reload both succeed from the produced artifact;
- the ordered 1,024-row prompt manifest is deterministic across two clean builds;
- every prompt receives exactly one terminal score under greedy decoding;
- evaluation accuracy is not at the preregistered floor or ceiling;
- measured export, conversion, reload, and evaluation time fit the later frozen
  per-run wall-time budget;
- all compact JSON is canonical and every large artifact is referenced by
  SHA-256.

Any failure closes this preflight attempt. There is no automatic retry. A repair
requires evidence, a new package identity, and separate authorization.

## Authority boundary

The eventual embedded authority must say, in machine-readable form:

- `training_allowed: false`
- `optimizer_steps_allowed: 0`
- `qualification_allowed: false`
- `pilot_allowed: false`
- `acquisition_allowed: false`
- `automatic_retry: false`
- `automatic_extension: false`

This file is not that authority and does not authorize an EOS launch.

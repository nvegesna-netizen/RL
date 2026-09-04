# Qwen3-1.7B GSM8K workload-transport no-training preflight

## Purpose

The next computation is a pinned, CPU-only/container preflight. It may resolve
configuration, load a small mocked GSM8K sample, run selected tests, and write a
canonical lock. It may not initialize distributed training, acquire GPUs, submit
an EOS job, qualify the workload, or collect scientific observations.

## Pinned environment

- image: `nemo-rl-nightly-5802754.sqsh`;
- image SHA-256:
  `3df8114a0b3e60ef95ce13f8b982c7cc45d164aca71c63a87388b1e8434ee470`;
- embedded source commit: `ae07eafe8035b5b2e84efa7234e70e7fd7e493c1`;
- source: an archive of the review-ready workload-transport commit, recorded by
  full commit ID and archive SHA-256 in the generated lock.

## Required checks

1. Resolve the exact workload overlay and validate every frozen model, GRPO,
   sampler, release-arm, dataset, verifier, output-path, and resource-cap field.
2. Validate the frozen protocol with the dedicated GSM8K workload-transport
   analyzer and demonstrate rejection by the OpenMath transport analyzer.
3. Verify the compatibility-audit and accepted OpenMath result-record hashes.
4. Exercise the new protocol and preflight tests plus the existing controlled
   release, opportunity ledger, adjusted inference, mechanism, and observer-duty
   tests selected by the manifest.
5. Exercise GSM8K parsing with mocked rows so the preflight does not depend on a
   mutable network dataset download.
6. Emit a canonical lock and summary containing the exact source, image,
   protocol, config, evidence, and selected-test hashes.

Any failure is terminal for this preflight attempt. There is no automatic retry
or extension. A passing no-training preflight does not authorize or count as a
qualification or scientific acquisition.

## Gate after success

After a clean preflight, review its preserved artifacts. Only then may a separate
32-step neutral qualification be proposed. Qualification data are operational
only and are forbidden from the causal estimator. The 558-step confirmatory
acquisition remains a later, separately gated action under the frozen four-hour,
two-GPU cap.

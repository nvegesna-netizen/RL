# Complete replay environment reconstruction

2026-10-14. The user requested full preparation and launch after two packaging
failures. Both prior guards and original results are preserved.

## Why these new errors occurred

The new extension was imported directly in a standalone driver process. Earlier
successful restoration runs did not do this; they let the worker factory select
the prefetched MCORE backend. The new package incorrectly used the base
interpreter and forced actors to inherit that interpreter. Path-only testing
then fixed Core visibility without addressing the complete backend environment.
These are implementation errors in the new package, not changes in the
scientific data or a four-hour timeout.

The successful restoration trace, EOS job `478520112`, shows both Torch and
Transformer Engine loading in
`/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker`.
It also shows Bridge importing from the image's backend checkout, while Core
imports from the qualified captured source. This is direct evidence that the
base interpreter was not an adequate substitute for the backend interpreter.

## Reconstructed package

Use that exact prefetched MCORE interpreter for **both** driver and actor.
The system-actor selector now deliberately refers to the MCORE driver, not
`/opt/nemo_rl_venv/bin/python`. Require the interpreter to exist; do not build
or silently switch environments at runtime.

Include the preserved Bridge archive, SHA-256
`1429945d1d50045900e40314fa283fa5f66484e945aad4920da137d7ad2c9313`,
whose PAX commit `573e088c9c6740082c39744e03dc5b009e730ed4` matches the
qualified source's Bridge gitlink and the successful restore source's gitlink.
Extract base, Bridge, then Core, and explicitly expose both source namespaces.
The Core archive, model image, checkpoint, and frozen scientific protocol are
unchanged. No dependency upgrade or fresh lockfile resolution is introduced.

## Stronger runtime gate

Before checkpoint retrieval, actually import Torch, Transformer Engine,
Core, Bridge checkpointing, TQPolicy, and the complete custom worker in the
driver. Verify the MCORE prefix, exact bundled source origins, and frozen
TransferQueue commit. Exercise the real state-digest implementation.

Repeat that check in a separate Ray initializer-like process using the same
explicit MCORE interpreter. Compare interpreter prefix, Torch version, source
origins and dependency identity. Only a successful combined result permits
transport and the unchanged two comparator replays. This checks imports and
environment identity, not full optimizer training; the subsequent qualification
still checks state restoration and exactly one update per fresh process.

Thirteen CPU checks pass on the exact clean-room extraction, including both
source namespaces and refusal to use the base interpreter. The container
cannot run on this Mac. A read-only SSH check found enroot on the configured
cluster login but the EOS image path is not mounted there, so no image-level
pass is claimed locally. The actual full import gate is executed in the EOS
workload before data retrieval. Expanded Bash, embedded Python, compilation,
lint, and JET manifest-model checks pass.

## Scope

One newly authorized guarded `runllm.py --no_wait` invocation. Four-hour
allocated runtime, no client queue deadline, no automatic retries. The
checkpoint remains outside uploaded assets. No Shield update or post-update
held-out loss is computed, no historical contrast is opened, and the fresh
20-seed cohort is not launched. The original outcome and complete-state
qualification gates are unchanged.

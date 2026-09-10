# Llama lifecycle-derived v2 bootstrap preflight

Status: `FROZEN_AUTHORIZED_PENDING_SUBMISSION`.

Exactly one credential-free, no-training EOS preflight is authorized. It must
use `runllm.py --no_wait`, the pinned v1 scientific source archive and container,
and the frozen v2 operational amendment. Automatic retry and extension are
disabled, and the workload is limited to 15 minutes on one node.

The workload may only:

1. authenticate and safely extract the unchanged source archive;
2. authenticate the v2 amendment, repair result, and launch authorization;
3. reproduce the container fingerprint check;
4. import `nemo_rl` immediately before `megatron`;
5. prove that `megatron.__path__` contains the vendored Megatron-LM tree; and
6. emit a compact result and hash ledger.

It may not invoke a GRPO entrypoint, download model weights, train, qualify,
acquire, retry, extend, or submit another workload. Success authorizes planning
of a fresh paired qualification package; it does not itself authorize that
qualification or any acquisition.

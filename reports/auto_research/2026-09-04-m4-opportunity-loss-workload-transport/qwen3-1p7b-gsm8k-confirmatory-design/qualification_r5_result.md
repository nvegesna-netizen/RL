# GSM8K neutral qualification R5 result

Status: `FAILED_MISSING_CONTAINER_SUBMODULE_BASELINE`; the scientific question
was not tested and no retry occurred.

The one authorized R5 release submitted the frozen R4 manifest through
`runllm.py --no_wait`. Parent pipeline `66314566` generated downstream pipeline
`66315280`. Generator `426524155`, logs-before `426529990`, and logs-after
`426529992` succeeded. EOS compute `426529991` ran as Slurm job `5977831` and
exited `1`, making both pipelines terminal `failed`.

All frozen package gates passed: safe archive extraction reported 1,910 members
and 16 symlinks, the preflight-delta comparison reported 31 unchanged and two
attested files, and the authoritative config validator passed. Both the
Megatron policy worker and vLLM worker initialized. The neutral instrument then
recorded 43 assignments, 244 sibling completions, and 27 complete opportunity
groups at start version zero. Corrected observer duty was 0.00221865, below the
registered 0.01 ceiling. No trainer step completed and no qualification summary
or causal estimate was produced.

The terminal log identifies the packaging defect directly. Every Ray process
reported the Automodel, Gym, and Megatron-Bridge submodules as `missing` from
the extracted run tree. When `SingleControllerActor` deserialized the first
training result returned by `MegatronPolicyWorker`, Python raised:

```text
ModuleNotFoundError: No module named 'megatron'
```

The source archive was created by `git archive`; it contains the submodule
mount-point directories but not submodule contents. R4 extracted it into an
empty directory and set that directory as `PYTHONPATH`. By contrast, the
accepted Qwen3-1.7B qualification, using the same pinned image and
`/opt/nemo_rl_venv/bin/python`, first copied `/opt/nemo-rl` into its isolated run
tree and then overlaid the frozen source archive. Its manifest SHA-256 is
`fbae4213355a3f4a21e86e8185e7451529f44e3d3a9be65c387a29364143764c`.

The proper repair therefore changes bootstrap packaging only: copy the pinned
container baseline, overlay the unchanged source archive, and reject the run
before model initialization unless the complete dependency fingerprint matches
the pinned container and `megatron` imports from the isolated run tree. It does
not change the model, dataset, qualification instrument, trainer steps,
estimator, source commit, or scientific protocol. A new R6 package contract
records this repair but authorizes no submission or acquisition.

The preserved 47,318,611-byte artifact remains outside Git at
`session/20260903_m4_qwen3_1p7b_transport/gsm8k-qualification-r5-66315280-job-426529991.zip`
with SHA-256
`43aff533273fa96b0a3df523307fe4b6a1730ebcd197819d91363598c9a0ef47`.

# GSM8K neutral qualification R2 result

Status: `FAILED_INVALID_DISABLED_INSTRUMENT_AUDIT_COMBINATION`; qualification
training did not start.

The single authorized R2 release submitted manifest SHA-256
`d1dccd7ed62f4aee6fd0a32e1351e228a03dcada9c8ab27a55503774949c6818`
with `runllm.py --no_wait`. Parent pipeline `66269570` generated downstream
pipeline `66269740`. Generator `426126476`, logs-before `426127526`, and
logs-after `426127528` succeeded. EOS compute `426127527` ran as Slurm job
`5976307` and exited `1`, making both pipelines terminal `failed`.

The R1 repair worked: safe extraction, frozen-evidence checks, config loading,
and the individual two-GPU topology assertions all passed. The authoritative
`validate_single_controller_config` call then rejected this combination:

```text
ValueError: async_rl.gradient_opportunity_audit.enabled=true requires
async_rl.controlled_release_delay.enabled=true
```

The opportunity audit uses the controlled-release instrument's assignment
domain and arm labels. The accepted Qwen3-1.7B neutral qualification therefore
enabled the instrument with one `neutral` arm of exactly `0.0` seconds; it
completed 32 steps and produced 544 opportunity groups without a causal
contrast. The GSM8K design instead disabled the instrument while requiring the
same audit, an internally inconsistent operational configuration.

The proper repair is not to weaken the core validator. It is to activate the
validated instrument with the sole zero-dose neutral arm, continue to prohibit
any control-versus-delay estimate, and require complete neutral assignment
events with zero delay. This preserves the qualification's non-causal purpose.

Failure occurred before `M4_GSM8K_NEUTRAL_QUALIFICATION_START`; no run log,
lifecycle or opportunity records, observer-duty result, qualification summary,
or causal estimate was produced. Scientific acquisition remains unauthorized.

The preserved 47,207,749-byte artifact remains outside Git at
`session/20260903_m4_qwen3_1p7b_transport/gsm8k-qualification-r2-66269740-job-426127527.zip`
with SHA-256
`428dc43c8b5c76e65dee40455d4b90ff26ad0da4da2ba38b62b1efa233330671`.
R2 authority is consumed; no retry occurred.

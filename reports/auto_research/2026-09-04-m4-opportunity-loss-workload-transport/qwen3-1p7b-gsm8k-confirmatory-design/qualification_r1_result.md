# GSM8K neutral qualification R1 result

Status: `FAILED_CONFIG_GATE_OVERSTRICT`; qualification training did not start.

The single authorized corrected release submitted manifest SHA-256
`cb0a23720d218251662fd4a22b054a6e298eec1d8f538c999631a91cf25940b3`
with `runllm.py --no_wait`. Parent pipeline `66264517` generated downstream
pipeline `66264953`. Logs-before job `426086492`, generator job `426085047`,
and logs-after job `426086496` succeeded. EOS compute job `426086494` ran as
Slurm job `5976209` and exited `1`, making both pipelines terminal `failed`.

The corrected JET formatting repair worked: downstream generation, the EOS
handoff, safe extraction of all 1,904 archive members, and every frozen-evidence
hash check passed. The next embedded config gate loaded and resolved the
qualification config, then failed its twelfth assertion:

```python
assert c["cluster"] == {"gpus_per_node": 2, "num_nodes": 1}
```

The inherited single-controller config also supplies
`master_port_range_low: 25000` and `master_port_range_high: 28000`. Those are
valid cluster-schema fields, but exact dictionary equality rejects them. The
gate should check the two intended topology values individually instead of
forbidding unrelated inherited fields.

This is an operational validator failure, not evidence about GSM8K resource
fit, reward support, observer duty, throughput, or the M4 hypothesis. It
occurred before `M4_GSM8K_NEUTRAL_QUALIFICATION_START`; the artifact contains
no qualification run log, lifecycle or opportunity records, observer-duty
result, terminal summary, or causal estimate. Scientific acquisition remains
unauthorized and did not start.

The preserved 47,206,577-byte compute artifact is outside Git at
`session/20260903_m4_qwen3_1p7b_transport/gsm8k-qualification-r1-66264953-job-426086494.zip`
with SHA-256
`8570351a38755e838e47c11ad0614b0865505328988260cd63b73e3bfad6d752`.
The R1 one-shot authorization is consumed; no retry occurred and no further
launch is authorized.

# Gradient-utility qualification generator failure

The first accepted JET qualification submission created upstream pipeline
`72126754` and generator job `473737737`. The generator failed with
`script_failure`; the pipeline produced no downstream pipeline and therefore no
EOS workload, Slurm allocation, GPU execution, gradient data, or scientific
outcome.

The generator did successfully materialize its diagnostic archive. That archive
contains only `custom_config.yaml` and `manifests/jet_workloads.yaml`. The
generated workload YAML is 16,750,190 bytes. This leaves only 27,026 bytes below
16 MiB before JET/GitLab adds the downstream CI wrapper. The trace then reaches
GitLab's 4 MiB log limit while printing the embedded payload and exposes no
runtime command: the failure is consistent with a downstream configuration
transport limit, not with source extraction, configuration resolution, tests,
model loading, or the audit path.

The repair changes only the credential-free source transport. The failed
package embedded a 10,468,973-byte full-repository archive of source commit
`702a40535`. The replacement embeds a 2,372,524-byte path-limited archive from
the same commit. It contains `nemo_rl/`, `examples/`, `tests/unit/`,
`pyproject.toml`, `uv.lock`, and the gradient-audit report directory. All
runtime-file hashes remain those frozen in the original qualification protocol.
The Megatron archive, container, custom EOS config, two-group qualification,
seeds, model, workload, no-update gates, runtime limit, and queue-deadline
suppression are unchanged.

This repair cannot alter a scientific result because no workload ran. A single
replacement submission is the narrowest test of the diagnosed packaging cause.


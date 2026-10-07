# Gradient-utility qualification generator failure

The first accepted JET qualification submission created upstream pipeline
`72126754` and generator job `473737737`. The generator failed with
`script_failure`; the pipeline produced no downstream pipeline and therefore no
EOS workload, Slurm allocation, GPU execution, gradient data, or scientific
outcome.

The generator did successfully materialize its diagnostic archive. That archive
contains only `custom_config.yaml` and `manifests/jet_workloads.yaml`. The
generated workload YAML is 16,750,190 bytes. Its proximity to 16 MiB motivated
an initial configuration-size hypothesis. That hypothesis is now falsified,
not retained as the diagnosis: a 5,958,188-byte replacement workload failed at
the same stage in pipeline `72128434`, generator job `473749648`.

The repair changes only the credential-free source transport. The failed
package embedded a 10,468,973-byte full-repository archive of source commit
`702a40535`. The replacement embeds a 2,372,524-byte path-limited archive from
the same commit. It contains `nemo_rl/`, `examples/`, `tests/unit/`,
`pyproject.toml`, `uv.lock`, and the gradient-audit report directory. All
runtime-file hashes remain those frozen in the original qualification protocol.
The Megatron archive, container, custom EOS config, two-group qualification,
seeds, model, workload, no-update gates, runtime limit, and queue-deadline
suppression are unchanged.

The second trace authenticates the common cause. JET API `3.129.1` raised
`RegistryLoadingError` while substituting the embedded script:
`NameError ... name 'names' is not defined`. The manifest builder escaped
Python, shell, and JSON braces through its own Python f-string, but emitted
single braces into the final workload. JET consequently interpreted expressions
such as the Python set comprehension as JET placeholders. A known-successful
generator artifact preserves those runtime braces as doubled braces in the
final manifest.

Neither attempt created a downstream pipeline, EOS workload, Slurm allocation,
GPU execution, gradient data, or scientific outcome. The correct repair is
therefore limited to JET-layer brace escaping. Source archive, source commit,
protocol, model, workload, seeds, no-update gates, runtime limit, and queue
behavior remain unchanged.

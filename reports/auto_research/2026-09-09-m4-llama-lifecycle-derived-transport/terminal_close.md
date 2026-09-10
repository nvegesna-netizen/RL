# Terminal close: Llama lifecycle-derived M4 successor

Status: `TERMINAL_CLOSED_PAIRED_QUALIFICATION_PACKAGE_FAILURE_NO_TRAINING_NO_ACQUISITION`.

## What passed

The separately preregistered lifecycle-derived instrument passed its one-shot
no-training preflight. Parent pipeline `67087159` and child pipeline `67087387`
were successful. The authenticated artifact records 67 selected tests, exact
equality of the 35-file local and remote locks, successful schema and static
checks, and metadata-only Llama access without downloading model weights.

## What failed

The two neutral qualifications were frozen and submitted together before either
result was inspected. OpenMath parent/child pipelines `67090802`/`67090910` and
GSM8K parent/child pipelines `67090748`/`67090857` failed identically at the
pre-training Megatron import check:

`ModuleNotFoundError: No module named 'megatron'`.

The package copied the pinned container tree and overlaid the frozen source, but
its embedded bootstrap check attempted `import megatron` directly. In this
repository, importing `nemo_rl` first appends the copied Megatron-LM workspace
to `sys.path`. The accepted qualification bootstrap explicitly used
`import nemo_rl` followed by `import megatron`; this package omitted the first
import. Both traces stop at that check before the training command. Both failed
workload artifacts contain empty metrics directories and no lifecycle,
opportunity, duty, derivation, or qualification-summary artifacts.

## Scientific interpretation

This is a packaging failure, not a scientific qualification miss. It provides
no evidence for or against the lifecycle-derived instrument, M4 opportunity
loss on Llama, or workload heterogeneity. No trainer step or causal acquisition
ran, so no Llama effect estimate exists.

The distinction does not change the registered stop rule. The protocol states
that either qualification failure closes the successor permanently without
retry, repair, threshold relaxation, extension, or one-cell substitution.
Therefore acquisition is ineligible and the successor is terminally closed.
A future attempt would require a new, separately preregistered study and must
not be described as a continuation or repair of this closed sequence.

## Preserved evidence

- OpenMath trace SHA-256:
  `86ac7f01b63e7ba1e157536b000b937efa61396f819c4dde50c6ed05a45c6c54`.
- GSM8K trace SHA-256:
  `6a973f0a88af5df63424d9ff78a9b21dd4423f01c17b9a27689e5318ca0e7a3e`.
- OpenMath workload artifact SHA-256:
  `50d53147193def2088ac95b09ffff264b144863b9baf6dc4469cef33a27d65d7`.
- GSM8K workload artifact SHA-256:
  `76e03ec0c63255831809b0c3f5466b3f72d0cd6732552ab4a0c4abf3e99847f4`.
- Compact machine-readable result: `qualification_terminal_result.json`.

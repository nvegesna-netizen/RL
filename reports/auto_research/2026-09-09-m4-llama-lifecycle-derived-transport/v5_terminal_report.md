# Llama M4 V5 terminal report

Status: `TERMINAL_MATERIAL_BOTH_WORKLOADS_REPLICATED`.

All four preregistered 448-step acquisitions completed successfully and passed
artifact authentication. Every run had zero terminal missingness in both arms,
the lifecycle-derived instrument was complete, and corrected recorder duty was
below 0.079%, versus the frozen 1% ceiling.

## Primary replicated results

| Workload | Replicate estimates | Equal-weight estimate | 95% confidence envelope | Conclusion |
| --- | --- | ---: | ---: | --- |
| OpenMathInstruct-2 | 0.3947, 0.4034 | 0.3991 | [0.3504, 0.4467] | `MATERIAL` |
| GSM8K | 0.3550, 0.3372 | 0.3461 | [0.2952, 0.3898] | `MATERIAL` |

Both lower confidence limits exceed the preregistered materiality threshold of
0.2. The conservative materiality p-value is `1/20001` in each workload. The
replicate-difference envelopes include zero for OpenMath
`[-0.1048, 0.0862]` and GSM8K `[-0.0852, 0.1069]`; there is no detected
within-workload replicate conflict.

The secondary OpenMath-minus-GSM8K estimate is 0.0530 with confidence envelope
`[-0.0137, 0.1232]`. Thus both workloads show material opportunity loss, but
the evidence does not establish that the effect differs between them.

## Interpretation

In the tested Llama 3.2 1B asynchronous GRPO setting, imposing the registered
five-second release delay caused a material increase in opportunity-weighted
loss relative to control on both OpenMathInstruct-2 and GSM8K. The result is
prospectively replicated within each workload and extends the M4 mechanism to a
different model family.

This does not establish a family-wide Llama effect, a pure architecture effect,
or generalization to other model sizes, delay magnitudes, training systems, or
workloads. The inconclusive cross-workload contrast must not be described as
evidence that the two workload effects are equal.

The full terminal analysis remains outside Git with SHA-256
`47b14b0caddae4ad055986eeb4d08a3dbaf5be6d4ab156f615503befe49e46b7`.
It was reproduced byte-for-byte from the authenticated artifacts.

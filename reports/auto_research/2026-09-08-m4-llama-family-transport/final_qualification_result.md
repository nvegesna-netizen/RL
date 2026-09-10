# Llama 3.2 1B final paired qualification result

Status: `TERMINAL_CLOSED_QUALIFICATION_MISS_NO_ACQUISITION`.

Both final neutral qualifications were inspected only after both pipeline graphs
were terminal. They used the same preflighted source commit, `d73b535db1eef0e5757fbf2950480903f6f2da62`, and neither run contained a
randomized causal treatment.

| Cell | Pipeline result | Steps | Groups | Corrected observer duty | Registered result |
| --- | --- | ---: | ---: | ---: | --- |
| Llama 3.2 1B / OpenMath | success | 32 | 756 | 0.00595336 | all eight gates pass |
| Llama 3.2 1B / GSM8K | failed after training | 32 | 1,089 | 0.01064760 | observer-duty gate fails |

OpenMath projects 9,450 assignments in the 400-version primary window, 0.835
wall-hours, and 1.669 two-GPU-hours. GSM8K projects 13,612.5 assignments,
0.663 wall-hours, and 1.327 two-GPU-hours. Both observed both binary reward
values, used only the neutral zero-second release arm, and have supported
bounded-shutdown lifecycle records. OpenMath had 16 assignments bounded before
release start; GSM8K had three. Neither had a bounded shutdown during release.

GSM8K's corrected observer duty is 0.01064760 against the unchanged 0.01
ceiling. Its mean corrected observer cost was 1.32254 ms per observation; the
run-specific 1% allowance was 1.24210 ms. The miss is therefore 0.06476
percentage points, or 6.48% relative to the ceiling. The workload completed all
32 steps and then exited at the registered `observer_duty` assertion, so this is
a genuine qualification-gate miss rather than an infrastructure, credential,
model-access, or post-run API failure.

The prospectively frozen stop rule states that either final qualification miss
permanently closes this Llama branch. Consequently there will be no retry,
threshold relaxation, further observer repair, single-cell substitution, or
448-step Llama acquisition. These qualification observations remain excluded
from every causal estimator and provide no evidence for or against an M4 causal
effect. The completed Qwen six-cell study and its claims are unchanged.

Compact pipeline, job, metric, and SHA-256 provenance is preserved in
`final_qualification_result.json`; large GitLab artifacts remain outside Git.

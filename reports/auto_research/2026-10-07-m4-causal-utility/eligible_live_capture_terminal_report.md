# Eligible-live decision capture qualification

Date: 2026-10-07

The outcome-excluded qualification passed. EOS job `474952005` and Slurm job
`6195702` completed with exit code `0:0`. The authenticated result is
`PASS_OUTCOME_EXCLUDED_ELIGIBLE_CAPTURE`: one eligible live M4-Shield decision
was captured at learner version 7, the Shield and reward-variance actions
differed, every frozen eligibility and integrity check passed, four disjoint
held-out groups were retained, and the checkpoint and tensor inventories
authenticated. The captured update was not executed and no post-update outcome
was opened.

The parent and child pipelines are red only because the separate `logs_after`
job attempted to download the successful workload's 13.46 GB artifact into a
Kubernetes pod with a 10 GiB ephemeral-storage limit. The successful EOS
artifact remains available, and its compact result, recursive inventory, and
observer-duty files were retrieved directly and hashed. This collection-layer
failure does not alter the qualification result.

The qualification does not estimate causal utility. The next registered gate
is a checkpoint-restoration and held-out-loss-evaluation qualification that
must execute no treatment or comparator update and expose no paired contrast.
Only after that gate passes may the fixed 20-seed acquisition be packaged.

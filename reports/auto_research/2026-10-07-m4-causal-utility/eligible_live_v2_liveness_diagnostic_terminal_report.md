# Eligible-live v2 liveness diagnostic terminal report

The diagnostic is scientifically terminal with
`PASS_NO_STALL_REPRODUCED_CAPTURE_EXCLUDED`. The EOS workload and Slurm job
completed with exit code zero. All 22 focused tests passed, and an eligible
M4-Shield capsule was captured at learner version 5 without executing the
captured update or opening any post-update outcome.

The red parent/child pipeline status is not the diagnostic endpoint. The
successful EOS job uploaded a 13.4 GB artifact archive containing a 14.8 GB
uncompressed checkpoint. The downstream `logs_after` Kubernetes pod attempted
to download that archive and exceeded its 10 GiB ephemeral-storage limit. This
occurred after the workload, terminal classifier, checksum ledger, and artifact
upload all completed successfully.

The collection failure was repaired without a rerun: the compact diagnostic
result, capture result, capsule manifest, lifecycle ledger, OARS-v2 ledger,
opportunity ledger, observer-duty record, focused-test log, and checksum ledger
were retrieved individually from EOS job `481678515`. Their SHA-256 values
match the workload-owned checksum ledger. The large checkpoint remains in the
successful EOS job artifact and was not redundantly downloaded locally.

At the captured decision, the FIFO baseline imminent L1 was
`6418.704961359501` and the M4-Shield selection was `6539.156526565552`, with
valid-token counts `6024` and `6051`, respectively. These are qualification
mechanism measurements, not a causal outcome or terminal-quality result.

No replacement diagnostic should be launched. A later paired replay or v2
causal acquisition must be separately frozen and must access the checkpoint
directly from the successful EOS artifact rather than routing it through a
size-limited `logs_after` pod.

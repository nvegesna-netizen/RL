# Qwen3-1.7B Confirmatory No-Training Preflight Result

The exactly-once preflight reached the intended two-GPU EOS job but finished
red. This is a validator implementation failure, not a failure of the frozen
scientific design or model topology.

The selected suite passed 126 checks before
`test_transport_config_resolves_and_lock_forbids_execution` failed. The
validator looked for `acquisition_authorized` under `gate`; the preserved
qualification result stores that field under `confirmatory_planning`. The
validator therefore rejected the same qualification record on which the
prospective design was based.

Pipeline `66133112`, job `424958292`, ended with `script_failure`. Its
47,106,955-byte artifact has SHA-256
`9cf4059a672921c661e00639e8cae7a37573d632bd0e4d356020e83ea0a92e91`.
No training or scientific acquisition began, and no automatic retry occurred.

The source fix changes only the qualification-field lookup. A new source
archive, manifest, and explicit preflight authority are required before one
replacement preflight may be submitted. The frozen protocol remains unchanged.

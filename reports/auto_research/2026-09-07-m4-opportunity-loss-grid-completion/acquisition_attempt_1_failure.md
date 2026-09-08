# Qwen3-0.6B/GSM8K acquisition attempt 1 failure

Status: `ACQUISITION_COMPLETE_POST_RUN_ANALYSIS_PACKAGING_FAILURE`.

Pipelines `66675448` and `66675506` are red because the Slurm script exited
one. This was not a training or scientific-acquisition failure. The run passed
its frozen authority checks, 150 tests, Ruff checks, and configuration checks,
then completed all 558 registered trainer steps. It preserved the lifecycle,
opportunity, and observer-duty streams.

The failure occurred after acquisition. The manifest invoked
`tools/opportunity_loss_pipeline.py`, whose generic entrypoint accepts only the
common-instrumentation-v1 protocol. It therefore rejected the registered
`m4-opportunity-loss-qwen3-0p6b-gsm8k-grid-completion-v1` identity with
`OpportunityLossPipelineError: registered pipeline requires v1 protocol`.

The already frozen `tools/opportunity_loss_workload_transport_pipeline.py`
explicitly registers this grid protocol, model, assignment domain and seed,
500-version primary window, adjusted estimator, and 20,000-draw inference
geometry. Running that analyzer against the immutable terminal ledgers required
no new observations, protocol change, GPU acquisition, retry, or extension and
recovered the registered result.

The terminal ZIP has 36 members, passes ZIP integrity, is 137,287,850 bytes,
and has SHA-256
`6c2ebeb2b2bf67df16be815f372ea87cde8eeff4d23e63291d8a98ac94659075`.
The recovered analysis result SHA-256 is
`26d1650143c502e5790f194ab7c703051820e47993c8fcd9206bbec2bd7f10c2`.

No reacquisition is scientifically warranted. The failure should be treated as
a terminal packaging defect with a deterministic analysis-only recovery.

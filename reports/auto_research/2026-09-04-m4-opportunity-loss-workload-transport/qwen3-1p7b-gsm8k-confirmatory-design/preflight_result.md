# Qwen3-1.7B GSM8K workload preflight result

Status: `FAILED_TEST_ASSERTION_WORDING`.

The exactly-once no-training preflight reached the intended EOS workload and
failed in the selected test suite. Parent pipeline `66249474` generated
downstream pipeline `66250208`; EOS compute job `425970567` ran as Slurm job
`5975802` with the exact workload identity
`basic/m4-qwen3-1p7b-gsm8k-workload-no-training-preflight dgxh100_eos 00 [2 dgxh100_eos]`.

The failure is isolated to the assertion text in
`test_gsm8k_protocol_is_rejected_by_openmath_transport_analyzer`. The old
OpenMath analyzer correctly rejected the GSM8K protocol with
`OpportunityLossPipelineError`, which is the behavior the test was intended to
prove. The test then incorrectly required the exception message to contain
`transport protocol`; the implementation's stable rejection message instead
said that the transport pipeline requires its Qwen3-1.7B v1 identity.

Before fail-fast termination, the safe archive gate passed and 111 selected
tests passed. One test failed. Configuration lock generation, Ruff, compilation,
and the terminal green marker were not reached. Training and scientific
acquisition did not start.

The preserved terminal artifact is 47,164,579 bytes with SHA-256
`c4ab76f5ec9987568820a71acaa7fabffb6f9a96f1b8fb3155e51c1098f7960d`.
The submitted source archive SHA-256 was
`a3eda46512856b0ed4b12b2a45b51ac4799184703d3d4e1b4165d99cb421127e`;
the manifest SHA-256 was
`6bd6efe2672c10e78457bf53fee894173db0334350d8966e97f4f0a7292a0f5a`.

The assertion has been corrected locally to require the intended exception
type without coupling protocol isolation to incidental error prose. Automatic
retry remains disabled. A replacement preflight requires a new committed source
archive, manifest, receipt, guard, and explicit release; this failed package
will not be reused.

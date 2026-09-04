# GSM8K neutral qualification R3 result

Status: `FAILED_STALE_PREFLIGHT_TEST_LOCK`; qualification training did not
start.

The single authorized R3 release submitted manifest SHA-256
`2358d3572117c1cf9c582b945c4bb6995a7eff1552a584bec3de6b6e3c21f046`
with `runllm.py --no_wait`. Parent pipeline `66297406` generated downstream
pipeline `66297538`. Generator `426363005`, logs-before `426364600`, and
logs-after `426364602` succeeded. EOS compute `426364601` ran as Slurm job
`5976892` and exited `1`, making both pipelines terminal `failed`.

The package was launched correctly. Its image, source, contract,
authorization, preflight lock, and preflight summary hashes all matched, and
safe extraction passed for all 1,910 source members. The next frozen-evidence
gate compared every file recorded by the earlier successful no-training
preflight against the repaired R3 source and rejected exactly one mismatch:

```text
tests/unit/tools/test_opportunity_loss_gsm8k_transport_protocol.py
preflight: sha256 d8f3e0df9bbfc59a9a225822e3e459f60e0fe91751e9e0efd95d1cf033d71044, size 5789
R3 source: sha256 51c82c09d0b729dd833d95e0fedad884d55779a2996c2771aff9d919464cb0e6, size 6928
```

That change is the R3 regression test added to exercise the authoritative
config validator with the supported one-arm, zero-second neutral instrument.
It is test-only and expected, but the runtime comparator had no explicit
amendment for it. The failure is therefore a stale provenance binding in the
qualification package, not evidence against the zero-dose repair or GSM8K
compatibility.

Failure occurred before `M4_GSM8K_QUALIFICATION_FROZEN_EVIDENCE_PASS`, config
validation, or `M4_GSM8K_NEUTRAL_QUALIFICATION_START`. No run log, lifecycle or
opportunity records, observer-duty result, qualification summary, or causal
estimate was produced. Scientific acquisition remains unauthorized.

A proper successor package must retain the historical preflight artifact
unaltered and add a fail-closed delta attestation binding the old and new source
archives and the sole expected locked-file change above. It must still reject
every other preflight-lock mismatch. R3 authority is consumed and cannot be
reused.

The preserved 47,220,492-byte artifact remains outside Git at
`session/20260903_m4_qwen3_1p7b_transport/gsm8k-qualification-r3-66297538-job-426364601.zip`
with SHA-256
`b26b3a64d672d0ab15bbaa1de9a7d2794d884996623d7dab1a4ee8ea272c6739`.
No retry occurred.

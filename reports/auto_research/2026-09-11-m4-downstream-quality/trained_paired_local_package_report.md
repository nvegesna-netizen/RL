# M4 downstream-quality acquisition package

Status: `PASS_LOCAL_PACKAGE_UNLAUNCHABLE`.

The frozen 16-block/32-run downstream-quality study now has one deterministic,
credential-free JET candidate per registered run. No candidate was submitted or
executed. No model weights were accessed, no optimizer was initialized, and no
training or scientific acquisition occurred.

## Exact package boundary

Each independent one-node, two-GPU, four-hour candidate binds:

- runtime source commit `e90e400eb2219284b1e0b1726bdfd8e878ed8620`;
- the frozen protocol and 32-run identity manifest;
- its unique block, regime, training seed, assignment seed/domain, config hash,
  and isolated result path;
- the V8-proven Megatron-LM and Megatron-Bridge dependency archives;
- 448-step SingleController training, exact terminal export, Megatron-to-HF
  conversion, vLLM reload, and the immutable 1,024-prompt OpenMath evaluation;
- compact binary prompt scores, systems/mechanism references, and artifact hashes.

The current embedded authority is deliberately non-executable. Its dynamic guard
terminates before source extraction, configuration loading, model access, or the
training entrypoint. A future acquisition cannot be launched by reusing these
files as-is; it requires a separately authorized authority record and a new set
of authorization-bound manifest identities.

## Clean-room evidence

The validator passed all 32 candidates and reproduced every candidate byte for
byte. It checked shell syntax, all seven embedded Python stages, the order and
hash of all seven embedded payloads, source-archive path safety, equivalence to
the already successful V8 JET manifest schema surface, and absence of nested
launch commands. It dynamically exercised the no-execution guard for every run
and rejected an authority mutation that enabled training.

Credential scanning covered the rendered candidates and 3,542 decoded regular
files across the source, Megatron-LM, and Megatron-Bridge archives. No embedded
token or private-key pattern was found. The 32 ignored manifests total
571,190,064 bytes; they remain outside Git. Their individual SHA-256 values and
byte counts are preserved in `trained_paired_local_package_validation.json`.

## Frozen identities

- Source archive SHA-256:
  `7e10c4252e7f20c877a02b95e3bf5df3937c71bfc8772da2da47350d8db75f64`
- Builder SHA-256:
  `0aa35fe2017bd9780c87e6910835ffaaba2c609afde4188fc100b7441b60f295`
- Protocol SHA-256:
  `dbe7a4f7d1938ef43d24e536ccf9dd57110a01bab5e6cb511980218e472ef2c0`
- Run-manifest SHA-256:
  `a461e83c7dbccdefa1c5c779062e709f34bfc502639a3e7de663a717fefa93bf`
- Package-authorization SHA-256:
  `7b5830d5347125ebcf73d6df6da86ad1b181ef634f230687a516d504cf0fdbc3`

## Next boundary

The package is ready for scientific acquisition authorization, not launch under
the present authority. Before any EOS call, freeze one authorization that names
all 32 identities, requires `runllm.py --no_wait`, permits model access,
optimizer initialization, training, and acquisition, preserves the four-hour
per-run and aggregate resource caps, requires submission without outcome
inspection, and forbids retry, replacement, and extension. Then construct and
clean-room validate new authorization-bound manifests before submitting anything.

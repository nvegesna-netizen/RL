# OARS contention repair

The repair passes its local implementation gate. Ordinary weight-FIFO remains
eager by default. The repaired qualification config sets a shared
`selection_candidate_watermark` of eight: weight-FIFO waits for eight ready,
in-window groups and then selects the oldest four exactly as before. The OARS
observer records a proposal over the same eight candidates but cannot actuate
it.

This placement is deliberate. The watermark belongs to weight-FIFO rather than
the OARS wrapper, so a later FIFO control arm can use the identical readiness
rule. A prospective causal comparison can therefore vary the four-group choice
policy while holding the eight-candidate timing rule fixed.

The implementation fails closed when the watermark is nonpositive, exceeds the
sampler's admission-window capacity, or exceeds OARS's candidate safety cap.
The repaired analyzer requires all 64 decisions to expose exactly eight
candidates and all 70 four-of-eight combinations, in addition to the original
FIFO identity, metadata, service-budget, skip, duty, latency, completion, and
runtime gates.

Validation completed locally:

- 34 focused sampler and OARS contracts passed;
- the preserved legacy analyzer and repaired analyzer suites passed;
- Ruff formatting and lint passed;
- targeted Pyrefly reported zero errors;
- Python compilation, YAML parsing, inherited-config resolution, and the
  protocol JSON gate passed.

This is an induced-contention systems repair, not evidence about OARS effects or
training quality. It does not erase the prior observation that natural eager
FIFO produced zero contended decisions. No EOS launch is authorized or made by
this repair.

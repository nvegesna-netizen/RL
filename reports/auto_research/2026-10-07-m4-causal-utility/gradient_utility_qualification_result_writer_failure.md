# Gradient-utility qualification result-writer failure

The test-selection-repaired qualification created upstream pipeline
`72203185`, child pipeline `72203320`, EOS job `474307265`, and Slurm job
`6190026`. All preflight gates passed, including seven gradient-summary tests,
three controller configuration tests, twelve MCore begin/abort tests, and two
offline analyzer tests.

The runtime command then completed its intended two-group no-update audit and
returned exit code `0`. The authenticated gradient ledger contains two unique
eight-sibling groups, finite exact norms, two 16,384-bin sketches per group,
and an abort acknowledgement for both groups. Its terminal record reports zero
`finish_train_step` calls, optimizer steps, scheduler steps, and learner-version
advances, with the canonical parameter hash unchanged.

The wrapper failed only while writing the compact qualification-result JSON:

```text
result_path.write_text(json.dumps(result,indent=2,sort_keys=True)+"
                                                                    ^
SyntaxError: unterminated string literal (detected at line 31)
```

The manifest builder interpreted `\n` while constructing the shell script,
placing a literal newline inside the embedded Python string. Consequently, the
formal result JSON and final SHA-256 inventory were not created, and the
qualification cannot be labeled `PASS_RUNTIME_QUALIFICATION` from this run
despite the successful runtime evidence. This remains a qualification run and
does not classify the scientific construct or establish scheduler quality.

The narrow replacement writes the newline as `chr(10)` and adds generation-time
syntax compilation for every embedded Python heredoc. It changes no source,
model, workload, seeds, audit behavior, selected tests, two-group limit,
zero-update gate, runtime limit, or queue behavior.

Authenticated terminal records:

- Main-job artifact ZIP SHA-256:
  `7291de779411dbf8fef3c452fd5d480134b98feb83c3a16fe0c28f37f8af768d`
- Logs-after artifact ZIP SHA-256:
  `bdd1fcf7fbc59729222b27a0c255e0b9c8a7c603b5f70fc6a05dc7c8b667e80d`
- Workload output SHA-256:
  `a732566b13a894fe4fd4eb3ebb66e1f680a2db512b378da4be69eb5c91094bc5`
- Slurm output SHA-256:
  `cbd339bf6b8d50d9dd4b878303c72a09e81192fdc735f1aec1def15f2144d5ed`
- Gradient ledger SHA-256:
  `e1f9139f3e148d4b2895f7f474ad7d70a12177122546c6e8e660b5fd2bc1acdf`
- Runtime log SHA-256:
  `79c983b56541e2acbb0525dc954d2c93e81863d208ef3743a12aadf565a0fbf1`

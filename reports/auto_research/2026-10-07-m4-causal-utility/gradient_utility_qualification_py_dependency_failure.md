# Gradient-utility qualification pytest `py` dependency failure

The isolated-test-runner qualification created upstream pipeline `72145636`,
child pipeline `72145728`, EOS job `473878051`, and Slurm job `6187600`.
Safe extraction, configuration authentication, all seven gradient-summary
tests, all three controller configuration tests, the MCore import gate, and
construction of the isolated pytest shim passed.

The MCore-interpreter test command then failed before test collection because
the shim omitted pytest's pure-Python `py` compatibility module:

```text
File "/workspace/m4-gradient-utility-pytest-shim/_pytest/compat.py", line 20, in <module>
  import py
ModuleNotFoundError: No module named 'py'
```

The training command did not start. No optimizer step, scheduler step,
learner-version advance, gradient ledger, qualification result, or scientific
outcome was produced. This is a pre-acquisition test-runner packaging failure.

The narrow replacement adds the image's single-file, pure-Python `py.py`
compatibility module to the already isolated pytest shim. The launcher rejects
native extensions, exposes the shim only to the MCore test process, performs no
installation or network access, and does not expose the base environment's full
site-packages directory to the MCore interpreter. Source commit, protocol,
model, workload, seeds, two-group limit, zero-update gates, runtime limit, and
queue behavior remain unchanged.

Authenticated terminal records:

- Main-job artifact ZIP SHA-256:
  `bef494abd36b220e1e224d44e9781c91e207681aa8d844ac2dc65cc1b582b865`
- Logs-after artifact ZIP SHA-256:
  `a36578f83364e3352728fb19371679c75c86854d9b41e40790fe6f6883b76ce3`
- Workload output SHA-256:
  `3bb7456da32f79043515535d5c88f8d5ad163f55977bebb015163bcb0484a126`
- Slurm output SHA-256:
  `b84c414611c833ca520b6b79fcd570f0830d8b4e68de1da768eec9154052f504`

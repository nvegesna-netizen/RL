# Replay startup audit and bounded repair plan

2026-10-15. Investigation only: no implementation edit, new manifest, guard
reset, launcher invocation, dependency installation, Slurm allocation or push.
Existing paper edits are unrelated and preserved.

## Finding

The full-environment replacement reached Ray startup after the complete driver
inspection returned successfully. Its preflight temporary directory caused a
108-byte Unix-domain socket address, exceeding Linux's 107-byte pathname limit.
The failure is deterministic for the recorded timestamp/PID, not a timeout,
checkpoint problem, training failure, or new scientific result.

Parent `73906237`, child `73906717`, EOS `487936844`, logs-after `487936846`,
Slurm `6252452`, node `eos0571`. Queue `13:54:01`; allocated `00:03:19`.
The companion terminal-failure JSON preserves direct artifact hashes and
confirms that preflight/test logs match across both terminal copies. No
transport log, replay log or qualification result exists in either copy.
Twelve unit tests passed; the base-interpreter rejection test was correctly
skipped in the actual MCORE interpreter.

## Independent corroboration

1. Read the archived source actually packaged for this run, not only the
   current worktree. Its `init_ray(log_dir=...)` forwards the absolute log
   directory as Ray's `_temp_dir` on both attach and local-start paths.
   Changing only `TMPDIR` would therefore not fix this explicit override.
2. Read the official Ray source matching the packaged source lock's version,
   `2.55.1`. Ray creates a timestamp/PID session directory, then a `sockets`
   directory and socket filename. Validation measures UTF-8 bytes, not Python
   character count. Filename collisions may add an integer suffix.
3. Executed the actual upstream `validate_socket_filepath` function extracted
   by AST, with a Linux-platform namespace. The logged 108-byte path is rejected;
   shorter candidates pass. No Ray process or GPU was needed for this check.
4. Independently exercised actual `AF_UNIX` binding on the configured Linux
   login host using a temporary directory with automatic cleanup. A 107-byte
   pathname bound successfully; 108 bytes raised `AF_UNIX path too long`.
   No job was allocated. This confirms the kernel boundary independently of
   the Python validator.

The source lock is not evidence of the installed container's exact Ray version.
The failed traceback and Linux experiment establish the relevant limit
directly. The next runtime receipt should explicitly record `ray.__version__`,
Python version and interpreter prefix; no version upgrade is justified by this
failure. The collected `jet_log.zip` contained one `jet_log.yaml`, and its
relevant readable entries did not establish the installed Ray version.

Official sources:

- [Ray socket validator](https://github.com/ray-project/ray/blob/ray-2.55.1/python/ray/_private/utils.py#L1578).
  Retrieved raw-file SHA-256:
  `0d94997901564f2b5500c4ad1b66840ae7a1c2abb4f35e6a7a9cf465da125941`.
- [Ray session and socket naming](https://github.com/ray-project/ray/blob/ray-2.55.1/python/ray/_private/node.py#L164).
- [Ray collision suffix construction](https://github.com/ray-project/ray/blob/ray-2.55.1/python/ray/_private/node.py#L744).

## Path-budget comparison

The observed case uses the exact logged timestamp and PID `3164759`. The
conservative case substitutes a ten-digit PID and adds `.999999` to the
socket filename. These are explicit test bounds, not a claim about every
possible future Ray naming scheme.

| Ray root | Observed-case bytes | Conservative-case bytes | Linux limit |
| --- | ---: | ---: | ---: |
| `/workspace/m4-full-environment-preflight-ray` | 108, rejected | 118, rejected | 107 |
| `/workspace/m4-duplicate-ray-a` or `-b` | 93, accepted | 103, accepted | 107 |
| `/workspace/m4-eligible-live-restore-ray-a` | 105, accepted | 115, rejected | 107 |
| `/tmp/m4pf` or `/tmp/m4ra` or `/tmp/m4rb` | 73, accepted | 83, accepted | 107 |
| `/tmp/m4pf-abcdefgh` or corresponding replay root | 82 | 92 | 107 |

Successful restore job `478520112` explicitly invoked the older, longer restore
root. Its success does not establish that every descriptive directory name is
safe: timestamp/PID and collision suffixes contribute to the complete address.
The existing duplicate replay roots do not reproduce the present failure under
either tested case; shortening them is preventive margin, not another observed
failure. Readable artifact names can stay unchanged.

## Why our previous preparation missed it

The custom duplicate-update path introduced a standalone driver and new
packaging, unlike the qualified restore's base driver/backend-worker split.
The first package omitted proper backend initialization; the first replacement
addressed Core only and omitted Bridge. The latest package corrected both
namespace sources and the driver/actor interpreter. Evidence that this repair
worked is control flow reaching `init_ray` *after* full module imports,
dependency identity checks and the state-digest smoke check returned.

The new startup gate then added a descriptive Ray root without a socket budget
check. The local preflight tests check base-interpreter rejection and strings
for the intended imports/actor gate; neither starts Ray or constructs its
complete socket address. Their pass was never a full runtime qualification,
but the launch preparation should have caught this platform constraint.

## Additional confirmed startup-hardening issue

`preflight.run` calls `init_ray` on line 86, before entering its cleanup `try`
on line 87. A startup exception bypasses its explicit `ray.shutdown()`.
I executed the actual function AST with fake imports and a failing initializer:
events were `driver_import_pass`, `init_ray_attempt`; `shutdown` was not called.
This is a cleanup coverage gap, not the cause of the long pathname. It does not
prove that processes leaked in the terminal job; Ray/process-exit cleanup can
still act. The same audit should cover setup failures before the replay
runner's current cleanup region, without masking the original exception.

## Repair sequence

1. Keep all scientific protocol bytes, backend archives, interpreter selection,
   full import checks and complete-state qualification requirements unchanged.
2. Separate short node-local Ray roots from descriptive output/artifact paths.
   Allocate a fresh short root for preflight and each replay, for example with
   `/tmp/m4pf-`, `/tmp/m4ra-`, `/tmp/m4rb-` prefixes and a bounded unique suffix.
   Check the actual resulting root; do not assume suffix size or silently fall
   back to a long directory. Avoid recursive deletion or broad cleanup.
3. Add a standard-library path-budget check used by all three startup paths and
   the exact generated-package validator. Reserve timestamp, ten-digit PID,
   socket-name and collision-suffix space; test encoded UTF-8 byte length.
   Fail locally before submission for unsafe names, and again before Ray init.
4. Include Ray initialization in explicit cleanup coverage. Test the exception
   path with a fake initializer, and retain the original exception. Confirm
   each replay still runs in a separate fresh Python process.
5. Expand regression tests to reject the exact failed path and multibyte paths,
   accept the selected unique roots with headroom, cover both replay labels,
   verify startup-failure cleanup, and inspect the exact generated shell rather
   than only checking required strings in a module.
6. Verify that the runtime preflight exercises the actual qualification module
   import as well as the custom worker import in the chosen interpreter. Keep
   the independent Ray process check, compare runtime identities and preserve
   compact stage records even on failure. Do not substitute a fake actor for
   the actual EOS import gate.
7. Clean-room extract the new package; run all tests, embedded-Python and Bash
   checks, hashes, JET schema and scoped diff checks. Preserve all consumed
   guards; any later replacement needs its own manifest identity and guard.
8. Only then submit one engineering qualification using `runllm.py --no_wait`,
   refreshed setup credentials, `time_limit=14400`, and no client queue cap.
   Authenticate parent, child and exact workload. Actual actor initialization,
   checkpoint transport and duplicate one-update replay must still pass before
   any fresh scientific cohort is launched.

## Verification boundary

Established: exact failure, matching terminal evidence, correct driver imports,
kernel boundary, upstream-validator reproduction, candidate path margins and
cleanup control-flow gap. Not established: successful isolated actor startup,
full optimizer/RNG replay or equality after one update. This investigation
does not turn those unexecuted stages into passes. The Mac cannot run the
Linux/CUDA image; the login-host socket diagnostic was not a container test.

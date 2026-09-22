# OARS-v2 quality-primary actuation gate

The acquisition gate is ready: reward variance and absolute M4 each completed
64 outcome-excluded learner updates with exact eight-candidate actuation and
passed every corrected systems gate.

| Scorer | Exact enacted decisions | Combined duty | Runtime | Gate |
|---|---:|---:|---:|---|
| Reward variance | 64/64 | 0.2088% | 380.24 s | PASS |
| Absolute M4 | 64/64 | 0.2048% | 430.58 s | PASS |

Both runs had four unique selected groups per decision, exact 70-combination
searches for every scorer, zero skips, zero fallbacks, complete liveness
accounting, and authenticated byte-identical terminal copies. No terminal
training-quality outcome was evaluated in either qualification.

Reward variance required an offline analyzer correction: its runtime completed
successfully, but the original gate incorrectly required zero outstanding
replacement credit and nonzero candidate excess at the fixed horizon. The
corrected invariant is `earned - consumed = outstanding`, with outstanding
credit bounded to zero or one; zero candidate excess is valid when it reconciles
between the scheduler ledger and lifecycle stream. Runtime behavior and
artifacts were not changed or rerun.

Pipeline status is operational provenance, not a scientific endpoint. The
frozen 18-block, 54-run acquisition has not been launched and no acquisition
outcome has been opened.

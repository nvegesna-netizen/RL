# Enacted-OARS qualification failure

The enacted-OARS qualification attempt is closed as **infrastructure-censored
before OARS execution**. It is not a failed OARS result.

Both terminal archives passed ZIP integrity. The main and logs-after copies of
the two declared artifacts were byte-identical, and their internal SHA-256
manifest had exact coverage. Only after this authentication was the result
opened. It reports `TRAINING_FAILURE`, exit code 1, no scientific outcome
acquisition, and no training-quality analysis.

The run failed while Ray created the `TransferQueueController` actor. The
branch's compatibility shim constructed a new per-actor pip environment and
asked pip to clone the pinned requirement
`TransferQueue @ git+https://github.com/Ascend/TransferQueue.git@b266d39`.
That clone failed with GitHub credential/transport errors. The controller was
therefore never created. No lifecycle, opportunity, OARS, or observer-duty
ledger exists, no scheduler decision was made, and no training update ran.

This diagnosis is consistent with the successful FIFO arm: the same pinned
software initialized TransferQueue and completed 64 decisions there. The
difference is an external dependency fetch at actor startup, not an observed
difference between FIFO and OARS.

## Protocol disposition

The frozen protocol says that a platform or code failure does not authorize a
replacement. This attempt is consumed, the confirmatory acquisition remains
locked, and no outcome interpretation or automatic retry is permitted.

The narrow repair candidate is already represented by repository commit
`1eb92eb20`: add a guarded `inherit_baked_single_node` actor-environment mode.
The qualification manifest specifies exactly one node, and TransferQueue was
already importable by the driver from the base environment. The guard rejects
zero or multiple live Ray nodes, so it removes only the redundant network pip
installation without weakening the multi-node safety boundary. Any use of this
repair must be frozen in a separately versioned amendment, tested, and
explicitly authorized before one replacement qualification is submitted.

Canonical machine-readable evidence is in
`qualification_act_failure_authentication.json` and
`qualification_act_failure_receipt.json`.

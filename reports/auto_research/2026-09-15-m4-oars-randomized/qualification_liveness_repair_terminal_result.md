# Enacted-OARS liveness-repair qualification result

The excluded enacted-OARS qualification arm passed every frozen systems gate.

- Parent pipeline `68650696`, child pipeline `68651075`, main job `446349100`,
  and logs-after job `446349119` all finished successfully.
- Both terminal ZIP archives passed integrity checks. The six declared workload
  artifacts match the internal SHA-256 manifest, and their main/logs-after
  copies are byte-identical.
- The run completed 64 learner updates and 64 enacted-OARS decisions. Every
  decision used exactly eight candidates, enumerated 70 four-group
  combinations, selected four unique groups, matched the recorded OARS
  proposal, respected the `1.02` token budget, and had complete metadata.
- The bounded replacement path was exercised once, recorded as one
  `oars_candidate_excess`, and the run remained live through update 64.
- Gradient-observer duty was `0.001862944842362997`; OARS decision duty was
  `0.00002692652768785273`. Both are below the preregistered `0.01` ceiling.
  Measured workload runtime was `390.801637919` seconds.

This was a systems-only qualification. Training quality was not analyzed, the
qualification artifacts are excluded from confirmatory estimates, and this
result does not establish an OARS scientific effect. Together with the earlier
authenticated FIFO qualification pass, it satisfies the technical
qualification gate for freezing the prospective ten-pair confirmatory package.
The confirmatory acquisition remains unlaunched and requires separate explicit
authorization.

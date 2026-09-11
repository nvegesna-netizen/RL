# Llama 3.2 3B M4 size-extension terminal report

Status: `TERMINAL_JOINT_MATERIALITY_NOT_CONFIRMED`.

## Result

All four preregistered acquisitions completed successfully and passed the
evidence-integrity, lifecycle-derivation, observer-duty, strict-join, and
missingness gates. OpenMath replicated a material M4 opportunity-loss effect at
3B: the equal-replicate estimate is 0.2621 with confidence envelope
[0.2215, 0.2994], one-sided materiality p = 0.0020, and Holm-adjusted p =
0.0040. GSM8K's estimate is 0.2135, but its envelope [0.1872, 0.2398] crosses
the frozen 0.20 materiality threshold; p = 0.1566. GSM8K is therefore
`INCONCLUSIVE`, not `NOT_MATERIAL`.

The prospectively required joint criterion was that both workloads conclude
`MATERIAL`. It was not met. This is a scientific result, not an operational
failure: all pipelines and jobs succeeded, each arm/replicate has zero
missingness, and 28,788 primary assignments entered the four 3B analyses.

## Size extension

The prespecified secondary fixed-configuration comparisons show smaller effects
at 3B than at 1B on both workloads. The 3B-minus-1B estimate is -0.1370 for
OpenMath, with confidence envelope [-0.1986, -0.0749] and simultaneous interval
[-0.2093, -0.0647]. For GSM8K it is -0.1326, with envelope
[-0.1834, -0.0750] and simultaneous interval [-0.1939, -0.0712]. Both
simultaneous intervals exclude zero.

The difference between those two size attenuations is only 0.0045, with envelope
[-0.0766, 0.0878], so workload-specific differences in attenuation remain
inconclusive. The study supports attenuation between these tested 1B and 3B
configurations; it does not identify a causal effect of parameter count or a
general scaling law.

## Provenance and closure

The 1B terminal result was reproduced from its four authenticated ledgers before
forming size contrasts. The 3B terminal analysis was rerun independently and
was byte-identical, SHA-256
`6855611a3a400d84f45003e26f8ef40c2fcee0d6d2c0edbd3a98ed27c78c661a`.
The compact machine-readable result is `terminal_result.json`; raw artifacts
remain outside Git and are bound by `terminal_authentication.json`.
Qualification data never entered a causal estimator. The authorized acquisition
is complete and terminal, with no retry, extension, threshold change, workload
substitution, or outcome-guided sample increase.

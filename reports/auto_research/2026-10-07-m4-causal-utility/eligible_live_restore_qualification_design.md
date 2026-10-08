# Eligible-live restore and held-out evaluation qualification

The eligible-live capture passed, but that result alone does not establish that
the 14.83 GB distributed checkpoint can be restored into a fresh trainer or that
the frozen held-out loss can be evaluated without mutation. This qualification
tests those two prerequisites before any paired causal acquisition.

The EOS workload will authenticate the capsule manifest from successful job
`474952005`, download every checkpoint member named by that manifest, and
download only held-out group files `08` through `11`. Frontier files `00`
through `07` are forbidden. Consequently, the qualification cannot calculate a
base-versus-Shield contrast.

Two sequential, independent Python processes each create a fresh one-GPU
Megatron trainer from the same checkpoint. Each process requests optimizer and
scheduler restoration, authenticates the restored model parameter hash, runs
one forward-only clipped-PG loss evaluation over the same 32 held-out samples,
and authenticates the parameter hash again. `eval_mode=true` is mandatory;
optimizer and scheduler steps are forbidden.

The gate passes only when all downloaded bytes authenticate, both pre- and
post-evaluation parameter hashes equal the captured hash, both losses are
finite, and the independent losses agree within the prospectively frozen
absolute and relative tolerance of `1e-6`. A pass qualifies the machinery for
packaging the already-frozen 20-seed paired acquisition. It is not itself a
causal effect estimate and does not authorize or execute that acquisition.

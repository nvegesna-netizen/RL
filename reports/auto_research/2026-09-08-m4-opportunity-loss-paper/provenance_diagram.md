# Provenance diagram

```mermaid
flowchart TD
    I[Accepted instrument commits] --> P[Frozen protocol + source commit]
    P --> F[No-training preflight and hash lock]
    F --> Q[Neutral resource qualification]
    Q --> A[One-shot prospective acquisition]
    A --> R[Immutable lifecycle and opportunity ledgers]
    R --> H[SHA-256 authentication + terminal scoring]
    H --> G[Registered cell / interaction analysis]
    G --> S[Retrospective dependency-aware Qwen 2×3 synthesis]
    G --> L[Prospective replicated Llama workload synthesis]
    S --> M[Claim ledger, robustness tables, figures, manuscript]
    L --> M

    X[Failed packaging or preflight attempts] -. documented, excluded from estimator .-> H
    N[No outcome-guided retry or extension] -. constraint .-> A
    Z[Large raw artifacts outside Git] -. referenced by SHA-256 .-> R

    classDef frozen fill:#e3f2fd,stroke:#1565c0;
    classDef evidence fill:#e8f5e9,stroke:#2e7d32;
    classDef retrospective fill:#fff3e0,stroke:#ef6c00;
    class P,F,Q,A frozen;
    class R,H,G evidence;
    class S,M retrospective;
```

The publication synthesis authenticates the raw ledgers against the preserved
four-cell grid, NuminaMath, and four Llama V5 terminal records before
reconstructing any estimate. It does not consume qualification observations,
add acquisitions, or change registered conclusions. The Llama endpoints remain
separate from the Qwen interaction model.

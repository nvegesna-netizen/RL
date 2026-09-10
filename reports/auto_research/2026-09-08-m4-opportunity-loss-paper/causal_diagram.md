# Causal diagram

```mermaid
flowchart LR
    M[Model scale] --> C[System cadence and workload execution]
    W[Workload] --> C
    W --> Q[Pre-release gradient opportunity Q]
    M --> Q
    A[Randomized release arm: control or d5] --> T[Controlled release delay]
    T --> V[Version advance / direct-chain break]
    C --> V
    Q --> L[Lost opportunity Q × undelivered]
    V --> D[Terminal delivery disposition]
    D --> L
    O[Observer instrumentation] -. measures .-> Q
    O -. measures .-> D

    classDef randomized fill:#dff3ff,stroke:#1769aa,stroke-width:2px;
    classDef outcome fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef context fill:#fff3e0,stroke:#ef6c00;
    class A,T randomized;
    class D,L outcome;
    class M,W,C context;
```

The release arm is assigned from the frozen assignment domain before release.
Opportunity \(Q\) is recorded before the delay and therefore cannot be caused by
the assignment. The estimand compares arm-specific lost-opportunity means,
normalized by pre-delay opportunity. Model and workload are fixed cell-level
contexts rather than randomized treatments; their interaction is consequently
a contrast among the six tested Qwen environments, not a family-wide causal
effect. The Llama extension repeats the randomized within-cell contrast in a
second family, but family itself is not randomized and must not be interpreted
as the causal exposure in this diagram.

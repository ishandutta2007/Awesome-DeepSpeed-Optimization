# DeepSpeed-MoE (Sparse Expert Orchestration)

Coordinating distributed expert routing and memory optimization.

## Mermaid Diagram
```mermaid
graph TD
    T[Tokens] --> G[Gate Router]
    G -->|Rank 0| E1[Expert A]
    G -->|Rank 1| E2[Expert B]
    E1 & E2 --> Combine[Token Collation]
```

## Detailed Description
- **Sparse MoE Routing:** Routes specific tokens to matching experts.
- **All-to-All collectives:** Optimizes latency-sensitive collective communication boundaries across MoE groups.

[Back to main README](../README.md)

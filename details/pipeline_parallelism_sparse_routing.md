# Pipeline Parallelism & Sparse Routing (DeepSpeed-MoE & PP)

Combining pipeline partition and expert-based sparse computation.

## Mermaid Diagram
```mermaid
graph LR
    subgraph Node 1
        L1[Layer 1-4]
    end
    subgraph Node 2
        L2[Layer 5-8] --> MoE[MoE Gate]
    end
    subgraph Node 3
        E1[Expert 1]
    end
    subgraph Node 4
        E2[Expert 2]
    end
    MoE -->|All-to-All| E1
    MoE -->|All-to-All| E2
```

## Detailed Description
- **Pipeline Parallelism (PP):** Parts the model layers sequentially across nodes, using 1D pipeline paths.
- **DeepSpeed-MoE:** Leverages Mixture-of-Experts (MoE) routing, replacing standard feedforward layers with sparse expert kernels that activate subset routing pathways depending on tokens.

[Back to main README](../README.md)

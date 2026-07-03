# Parameter Sharding & Heterogeneous Memory (ZeRO-Stage 3 & ZeRO-Offload)

ZeRO-Stage 3 shards model parameters, while ZeRO-Offload utilizes host CPU and NVMe storage.

## Mermaid Diagram
```mermaid
graph TD
    subgraph GPU Memory
        P_shard[Parameter Shard]
    end
    subgraph CPU RAM
        Opt_state[Optimizer State]
    end
    subgraph NVMe
        Weights_off[Weights Swapping]
    end
    GPU Memory <-->|PCIe Bus| CPU RAM
    CPU RAM <-->|NVMe Bus| NVMe
```

## Detailed Description
- **ZeRO-Stage 3:** Completely partitions all model parameters across the GPUs. During the forward and backward passes, the parameters are gathered via communication primitives (`All-Gather`) and discarded immediately after use.
- **ZeRO-Offload:** Moves optimizer states and/or gradients to the CPU RAM/NVMe storage to free up VRAM, utilizing CPU computing resources to update optimizer states.

[Back to main README](../README.md)

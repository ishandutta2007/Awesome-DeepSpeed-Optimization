# The Interconnect All-to-All Network Bandwidth Wall

Mitigating communication constraints in distributed MoE & ZeRO-3.

## Mermaid Diagram
```mermaid
graph LR
    GPU1[GPU Nodes] <-->|InfiniBand / NVLink| GPU2[GPU Nodes]
    style GPU1 fill:#ff9,stroke:#333
    style GPU2 fill:#ff9,stroke:#333
```

## Detailed Description
- **Gradient Bucket Accumulation:** Groups small updates to maximize communication packet size.
- **Overlapping Collectives:** Interleaves compute with communication.

[Back to main README](../README.md)

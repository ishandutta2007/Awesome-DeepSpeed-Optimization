# ZeRO-Offload / ZeRO-Infinity (Heterogeneous Storage Swapping)

Exploits CPU and NVMe memory hierarchies for extreme scale training.

## Mermaid Diagram
```mermaid
graph LR
    GPU[GPU VRAM] <-->|ZeRO-Offload / Infinity| CPU[CPU RAM]
    CPU <-->|NVMe Engine| NVMe[NVMe SSD]
```

## Detailed Description
- **Memory Tiering:** Tracks parameter memory access patterns to prefetch weights from CPU RAM or NVMe arrays.
- **Infinity Engine:** Introduces parallel data transfers over NVMe and CPU boundaries to keep GPU computing units busy.

[Back to main README](../README.md)

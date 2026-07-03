# Overlap Communication Kernels

Asynchronous communication schedulers overlapping transfers and math.

## Mermaid Diagram
```mermaid
sequenceDiagram
    participant GPU Compute
    participant Communication Engine
    GPU Compute->>GPU Compute: Forward Pass (Layer L)
    activate GPU Compute
    Communication Engine->>Communication Engine: Asynchronous All-Gather (Layer L+1)
    deactivate GPU Compute
```

## Detailed Description
- **Asynchronous Execution:** Launches communication streams in parallel with CUDA compute kernels.
- **Double Buffering:** Allocates temporary buffers to hold incoming weights before swapping them in.

[Back to main README](../README.md)

# DeepSpeed-Inference (Fused Compression Serving)

Handwritten CUDA kernels fusing linear layer operators for fast inference.

## Mermaid Diagram
```mermaid
graph LR
    Linear[Linear Project] --> LayerNorm[Layer Norm] --> Act[Activation]
    style Linear fill:#f9f,stroke:#333
    style LayerNorm fill:#f9f,stroke:#333
    style Act fill:#f9f,stroke:#333
    Fused[Fused Single GPU Kernel]
```

## Detailed Description
- **Kernel Fusion:** Fuses multiple operations to minimize memory roundtrips to HBM.
- **Quantization Serving:** Integrates low-bit quantization kernels to double serving speeds.

[Back to main README](../README.md)

# The FP16 Precision Master Weight Underflow Hazard

Maintaining numerical precision during mixed-precision training.

## Mermaid Diagram
```mermaid
graph TD
    FP16_Grad[FP16 Gradients] --> Update[Update step]
    FP32_Master[FP32 Master Weights] --> Update
    Update -->|Cast| FP16_Params[FP16 Parameters]
```

## Detailed Description
- **FP32 Master Weights:** Updates model states using 32-bit float storage to avoid numerical decay.
- **Loss Scaling:** Dynamically scales loss values to protect small gradient calculations.

[Back to main README](../README.md)

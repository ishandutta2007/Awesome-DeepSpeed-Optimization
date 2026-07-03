# Sparsely Routed Mixture-of-Experts Scaling

Scale-up strategies for routing sparse networks.

## Mermaid Diagram
```mermaid
graph TD
    Tokens[Tokens] --> Router[Router Model]
    Router -->|Expert 1| Ex1[Expert 1]
    Router -->|Expert 2| Ex2[Expert 2]
    Router -->|Expert N| ExN[Expert N]
```

## Detailed Description
- **Load Balancing:** Enforces routing limits to prevent hot-spot experts from stalling nodes.
- **All-to-All Optimization:** Minimizes communication paths across distributed expert ranks.

[Back to main README](../README.md)

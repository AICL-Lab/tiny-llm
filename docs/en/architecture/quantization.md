# W8A16 Quantization

Deep dive into the weight-only INT8 quantization implementation in Tiny-LLM.

## Overview

W8A16 (Weight-8-bit, Activation-16-bit) is a quantization scheme that stores weights in INT8 format while keeping activations in FP16. This provides significant memory savings while maintaining inference quality.

```mermaid
flowchart LR
    subgraph FP16["FP16 Model"]
        W1["Weights: FP16<br/>16 bits/param"]
    end

    subgraph W8A16["W8A16 Quantized"]
        W2["Weights: INT8<br/>8 bits/param"]
        S["Scales: FP16<br/>1 per 128 params"]
    end

    FP16 -->|"~50% reduction"| W8A16

    style FP16 fill:#ef4444,stroke:#dc2626,color:#fff
    style W8A16 fill:#00D4AA,stroke:#00C49A,color:#fff
```

---

## Quantization Scheme

### Group-wise Quantization

Weights are divided into groups of 128 elements along the input dimension. Each group shares a scale factor.

```mermaid
flowchart TB
    subgraph WeightMatrix["Weight Matrix [rows, cols]"]
        G0["Group 0<br/>128 elements"]
        G1["Group 1<br/>128 elements"]
        G2["..."]
        GN["Group N<br/>128 elements"]
    end

    subgraph Scales["Scale Factors [rows/128, cols]"]
        S0["scale[0]"]
        S1["scale[1]"]
        S2["..."]
        SN["scale[N]"]
    end

    G0 --> S0
    G1 --> S1
    GN --> SN

    style WeightMatrix fill:#8B5CF6,stroke:#7c3aed,color:#fff
    style Scales fill:#F59E0B,stroke:#d97706,color:#fff
```

### Mathematical Formulation

For a weight matrix $W \in \mathbb{R}^{M \times N}$:

$$W_q = \text{clamp}\left(\text{round}\left(\frac{W}{s}\right), -128, 127\right)$$

Where $s$ is the per-group scale factor:

$$s_g = \frac{\max(|W_g|)}{127}$$

Dequantization for computation:

$$\hat{W}_g = W_{q,g} \times s_g$$

---

## Memory Comparison

```mermaid
xychart-beta
    title "Memory Usage for 7B Model"
    x-axis ["FP16", "FP8", "W8A16", "W4A16"]
    y-axis "Memory (GB)" 0-->16
    bar [14.0, 7.0, 7.2, 3.6]
```

| Quantization | Weights | Activations | Total Memory | Quality Loss |
|--------------|---------|-------------|--------------|--------------|
| FP16 | 14.0 GB | FP16 | 14.0 GB | Baseline |
| FP8 | 7.0 GB | FP8 | 7.0 GB | Moderate |
| **W8A16** | 7.2 GB | FP16 | 7.2 GB | **Minimal** |
| W4A16 | 3.6 GB | FP16 | 3.6 GB | Noticeable |

---

## Kernel Implementation

### W8A16 Matrix Multiplication

```cpp
void w8a16_matmul(
    const half* input,      // [M, K] FP16
    const int8_t* weight,   // [K, N] INT8
    const half* scales,     // [K/128, N] FP16
    half* output,           // [M, N] FP16
    int M, int N, int K,
    int group_size = 128,
    cudaStream_t stream = 0);
```

### Optimization Techniques

```mermaid
flowchart TB
    subgraph Optimizations["Kernel Optimizations"]
        A["Shared Memory Tiling"]
        B["Vectorized Loads"]
        C["Warp Shuffle"]
        D["Coalesced Writes"]
    end

    subgraph Benefits["Benefits"]
        BA["Reduced global memory access"]
        BB["4x load bandwidth"]
        BC["No shared memory bank conflicts"]
        BD["Maximum memory throughput"]
    end

    A --> BA
    B --> BB
    C --> BC
    D --> BD

    style Optimizations fill:#76B900,stroke:#5a8f00,color:#fff
    style Benefits fill:#00D4AA,stroke:#00C49A,color:#fff
```

#### Shared Memory Tiling

```cpp
// Load input tile to shared memory
__shared__ half input_tile[TILE_M][TILE_K];

// Vectorized load (128-bit = 8 FP16 values)
float4 vec = *reinterpret_cast<const float4*>(input + offset);
```

#### Warp Shuffle for Scale Broadcast

```cpp
// Broadcast scale to all threads in warp
float scale = __shfl_sync(0xffffffff, scales[scale_idx], 0);
```

---

## Quality Preservation

### Perplexity Comparison

```mermaid
xychart-beta
    title "Perplexity on WikiText-2 (Lower is Better)"
    x-axis ["FP16", "W8A16", "W4A16-GPTQ"]
    y-axis "Perplexity" 5-->6
    bar [5.47, 5.52, 5.68]
```

### Why W8A16 Maintains Quality

1. **No Activation Quantization**: FP16 activations preserve precision in the computation chain
2. **Per-group Scaling**: Fine-grained scales adapt to weight distribution
3. **Tensor Core INT8**: Hardware-accelerated dequantization during computation

---

## Integration with Transformer Layers

```mermaid
flowchart LR
    subgraph Layer["Transformer Layer"]
        Input["Hidden States<br/>FP16"]
        QKV["QKV Projection<br/>W8A16 MatMul"]
        Attn["Attention<br/>FP16"]
        Out["Output Proj<br/>W8A16 MatMul"]
        FFN["FFN<br/>W8A16 MatMul"]
    end

    Input --> QKV
    QKV --> Attn
    Attn --> Out
    Out --> FFN

    style QKV fill:#8B5CF6,stroke:#7c3aed,color:#fff
    style Out fill:#8B5CF6,stroke:#7c3aed,color:#fff
    style FFN fill:#8B5CF6,stroke:#7c3aed,color:#fff
```

---

## API Usage

### Loading Quantized Weights

```cpp
#include <tiny_llm/quantization.h>
#include <tiny_llm/model_loader.h>

// Load quantized weights from file
QuantizedWeight qweight;
qweight.rows = hidden_dim;
qweight.cols = hidden_dim;
qweight.group_size = 128;

// Allocate GPU memory
cudaMalloc(&qweight.data, qweight.weightBytes());
cudaMalloc(&qweight.scales, qweight.scaleBytes());

// Load from model file
model_loader.load_quantized_weight("q_proj", &qweight);
```

### Performing Quantized MatMul

```cpp
#include <tiny_llm/cuda_utils.h>

// W8A16 matrix multiplication
w8a16_matmul(
    input_fp16,      // [batch, seq, hidden]
    qweight.data,    // [hidden, hidden] INT8
    qweight.scales,  // [hidden/128, hidden] FP16
    output_fp16,     // [batch, seq, hidden]
    batch * seq_len, // M
    hidden_dim,      // N
    hidden_dim,      // K
    128,             // group_size
    stream
);
```

---

## References

- [LLM.int8(): 8-bit Matrix Multiplication for Neural Networks at Scale](https://arxiv.org/abs/2208.07339) - Dettmers et al., NeurIPS 2022
- [GPTQ: Accurate Post-Training Quantization](https://arxiv.org/abs/2210.17323) - Frantar et al., ICLR 2023

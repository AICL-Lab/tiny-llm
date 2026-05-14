# Architecture Overview

System architecture and design documentation for Tiny-LLM inference engine.

## Overview

Tiny-LLM is a high-performance CUDA C++ inference engine designed for efficient Transformer model inference. It focuses on:

| Feature | Technology | Benefit |
|---------|------------|---------|
| **W8A16 Quantization** | INT8 weights + FP16 activations | ~50% memory reduction |
| **Efficient KV Cache** | Incremental decoding with sequence management | O(1) autoregressive step |
| **Optimized Kernels** | Tensor Core INT8, shared memory tiling | Maximum throughput |
| **Modular Design** | Clean separation of concerns | Easy to extend and test |

---

## System Architecture

```mermaid
flowchart TB
    subgraph Input["📥 Input"]
        A["Model File<br/>.gguf/.bin"] --> B["GGUF Parser"]
        P["Prompt Tokens"] --> C["Tokenization"]
    end

    subgraph Engine["⚙️ InferenceEngine"]
        B --> D["Model Loader"]
        D --> E["InferenceEngine"]
        C --> E
        E --> F["Transformer Stack"]
    end

    subgraph Transformer["🔄 Transformer Layers"]
        F --> G["Embedding Layer"]
        G --> H["Layer 1..N"]
        H --> I["LM Head"]
    end

    subgraph KVCache["💾 KV Cache"]
        H <--> J["KV Cache Manager"]
        J --> K["Pre-allocated Slots"]
    end

    subgraph Output["📤 Output"]
        I --> L["Sampling"]
        L --> M["Generated Tokens"]
    end

    style A fill:#00D4AA,stroke:#00C49A,color:#fff
    style E fill:#76B900,stroke:#5a8f00,color:#fff
    style J fill:#F59E0B,stroke:#d97706,color:#fff
    style L fill:#8B5CF6,stroke:#7c3aed,color:#fff
```

---

## Core Components

### 1. InferenceEngine

The main entry point for model inference.

```cpp
class InferenceEngine {
public:
    // Load model from disk
    static Result\<std::unique_ptr\<InferenceEngine\>\> load(
        const std::string& path, const ModelConfig& config);
    
    // Complete generation pipeline
    std::vector<int> generate(
        const std::vector<int>& prompt, 
        const GenerationConfig& config);
    
    // Statistics and profiling
    const GenerationStats& getStats() const;
    void resetStats();
};
```

**Key Responsibilities**:
- Model lifecycle management
- Prefill/decode orchestration
- Token sampling and generation loop
- Performance profiling

### 2. KV Cache Manager

Efficient key-value cache for autoregressive generation.

```mermaid
flowchart LR
    subgraph CacheStructure["KV Cache Memory Layout"]
        K["K Cache<br/>[batch, layers, seq, heads, dim]"]
        V["V Cache<br/>[batch, layers, seq, heads, dim]"]
    end

    subgraph Operations["Operations"]
        A1["allocateSequence()"]
        A2["appendKV()"]
        A3["getCache()"]
        A4["advanceSeqLen()"]
    end

    A1 --> CacheStructure
    A2 --> CacheStructure
    A3 --> CacheStructure
    A4 --> CacheStructure

    style K fill:#76B900,stroke:#5a8f00,color:#fff
    style V fill:#76B900,stroke:#5a8f00,color:#fff
```

### 3. W8A16 Quantization

Weight-only INT8 quantization with FP16 activations.

```mermaid
flowchart TB
    subgraph QuantScheme["Quantization Scheme"]
        W["Weight: INT8<br/>[rows, cols]"]
        S["Scales: FP16<br/>[rows/128, cols]"]
        O["Output: FP16<br/>= dequant(W, S) @ Act"]
    end

    W --> O
    S --> O

    subgraph GroupWise["Group-wise Layout"]
        G0["group 0 (128 elem) → scales[0]"]
        G1["group 1 (128 elem) → scales[1]"]
        GN["..."]
    end

    style W fill:#8B5CF6,stroke:#7c3aed,color:#fff
    style S fill:#F59E0B,stroke:#d97706,color:#fff
    style O fill:#00D4AA,stroke:#00C49A,color:#fff
```

**Benefits**:
- 50% weight memory reduction
- No activation quantization (maintains precision)
- Efficient INT8 Tensor Core utilization on Ampere+

---

## Data Flow

### Prefill Phase (Prompt Processing)

```mermaid
sequenceDiagram
    participant User
    participant Engine as InferenceEngine
    participant Trans as Transformer
    participant KV as KV Cache
    participant Sample as Sampling

    User->>Engine: generate(prompt, config)
    Engine->>Engine: tokenize(prompt)
    Engine->>Trans: forward(tokens, prefill=true)
    
    loop For each layer
        Trans->>Trans: RMSNorm
        Trans->>Trans: QKV Projection (W8A16)
        Trans->>Trans: RoPE
        Trans->>Trans: Attention Prefill
        Trans->>KV: store_kv(layer, k, v)
        Trans->>Trans: Output Projection
        Trans->>Trans: Residual
    end
    
    Trans-->>Engine: logits
    Engine->>Sample: sample(logits)
    Sample-->>Engine: first_token
    Engine-->>User: first_token
```

### Decode Phase (Token Generation)

```mermaid
sequenceDiagram
    participant User
    participant Engine as InferenceEngine
    participant Trans as Transformer
    participant KV as KV Cache
    participant Sample as Sampling

    loop For each new token
        User->>Engine: continue generation
        Engine->>Trans: forward(next_token, prefill=false)
        
        loop For each layer
            Trans->>Trans: RMSNorm
            Trans->>Trans: QKV Projection (W8A16)
            Trans->>Trans: RoPE
            Trans->>KV: get_cache(layer)
            Trans->>Trans: Attention Decode
            Trans->>KV: append_kv(layer, k, v)
            Trans->>Trans: Output Projection
            Trans->>Trans: Residual
        end
        
        Trans-->>Engine: logits
        Engine->>Sample: sample(logits)
        Sample-->>Engine: next_token
    end
    
    Engine-->>User: full_output
```

---

## Memory Layout

### Weight Storage

```mermaid
block-beta
    columns 1
    
    block:Header["Model Weights"]
        A["Token Embeddings<br/>FP16 [vocab, hidden]"]
    end
    
    block:Layers["Layer Weights (×N)"]
        B1["Attention<br/>INT8 q/k/v/o_proj"]
        B2["FFN<br/>INT8 gate/up/down"]
        B3["Scales<br/>FP16 per-group"]
    end
    
    block:Output["Output"]
        C["LM Head<br/>FP16 [hidden, vocab]"]
    end
```

### Activation Buffers

| Buffer | Shape | Dtype | Size (B=1, S=2048, H=4096) |
|--------|-------|-------|---------------------------|
| Hidden States | [B, S, H] | FP16 | 16 MB |
| Attention Output | [B, heads, S, head_dim] | FP16 | 16 MB |
| QKV | [B, S, 3×H] | FP16 | 48 MB |
| FFN Intermediate | [B, S, intermediate_dim] | FP16 | 44 MB |

---

## Performance Optimizations

### Memory Optimizations

| Technique | Implementation | Benefit |
|-----------|----------------|---------|
| W8A16 Quantization | Per-group INT8 weights + FP16 scales | 50% weight memory |
| KV Cache Paging | Pre-allocated with sequence management | Efficient batching |
| Activation Reuse | In-place operations | Reduced allocations |

### Compute Optimizations

| Technique | Application | Speedup |
|-----------|-------------|---------|
| Tensor Cores | INT8 matmul on Ampere+ | 2-4× vs FP16 |
| Warp Shuffle | Reductions | Eliminates shared memory |
| Vectorized Loads | 128-bit memory access | Better bandwidth |
| Kernel Fusion | RMSNorm+Resid, SiLU+Mul | Reduced kernel launch |

### Optimized Kernel Performance

```mermaid
xychart-beta
    title "Kernel Throughput (7B Model, A100)"
    x-axis ["w8a16_matmul", "attn_decode", "attn_prefill", "rmsnorm", "rope"]
    y-axis "Throughput" 0-->100
    bar [80, 65, 75, 95, 98]
```

---

## Design Principles

```mermaid
mindmap
  root((Design Principles))
    Modularity
      Clear interfaces
      Separation of concerns
      Plugin architecture
    Type Safety
      Result&lt;T&gt; errors
      Strong typing
      No exceptions
    RAII
      Auto GPU memory
      Stream management
      Resource cleanup
    Testability
      Unit tests
      Property-based
      CUDA testing
    Extensibility
      New kernels
      Sampling strategies
      Model formats
```

---

## Next Steps

- [Quantization Details](./quantization) - Deep dive into W8A16 implementation
- [KV Cache](./kv-cache) - Cache management strategies
- [CUDA Kernels](./cuda-kernels) - Kernel optimization techniques
- [API Reference](../api/inference-engine) - Complete API documentation

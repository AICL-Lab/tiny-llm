# Architecture Overview

Tiny-LLM is designed with simplicity and educational value in mind while maintaining production-quality performance.

## Core Components

```
┌─────────────────────────────────────────────────────┐
│                  InferenceEngine                     │
│                    (Public API)                      │
├─────────────────────────────────────────────────────┤
│                                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │ ModelConfig │  │   Result\<T\> │  │  KVCache    │ │
│  │             │  │             │  │  Manager    │ │
│  └─────────────┘  └─────────────┘  └─────────────┘ │
│                                                      │
├─────────────────────────────────────────────────────┤
│               TransformerLayer                       │
│  ┌─────────────────┐    ┌─────────────────┐        │
│  │  Attention      │    │      FFN        │        │
│  │  (W8A16)        │    │    (W8A16)      │        │
│  └─────────────────┘    └─────────────────┘        │
├─────────────────────────────────────────────────────┤
│               QuantizedWeight                        │
│         (INT8 weights + per-group scales)           │
├─────────────────────────────────────────────────────┤
│                  CUDA Kernels                        │
│   (Attention, FFN, Quantization, Memory Ops)        │
└─────────────────────────────────────────────────────┘
```

## Component Responsibilities

### InferenceEngine

The public API that orchestrates model loading and generation:

- `load()` - Load model weights from disk
- `generate()` - Generate text from a prompt
- `encode()` - Tokenize input text

### Result\<T\>

No-exception error propagation mechanism:

```cpp
Result<void> result = engine.load("model.bin");
if (!result.ok()) {
    // Handle error
    std::cerr << result.error() << std::endl;
}
```

### ModelConfig

Holds model hyperparameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `vocab_size` | `size_t` | Vocabulary size |
| `hidden_dim` | `size_t` | Hidden dimension |
| `num_layers` | `size_t` | Number of transformer layers |
| `num_heads` | `size_t` | Number of attention heads |
| `max_seq_len` | `size_t` | Maximum sequence length |

### QuantizedWeight

INT8 quantized weights with per-group scales:

- Weights stored as INT8
- Per-group scaling factors for dequantization
- Supports W8A16 inference pattern

### TransformerLayer

Single transformer layer with:

- Multi-head self-attention
- Feed-forward network
- Layer normalization
- Residual connections

### KVCacheManager

Manages key-value cache for efficient generation:

- Pre-allocated cache slots
- Per-sequence cache management
- Memory-efficient caching strategy

## Memory Layout

```
┌────────────────────────────────────────┐
│           Weight Memory                │
│  ┌──────────────────────────────────┐  │
│  │   Embedding Table (FP16)         │  │
│  └──────────────────────────────────┘  │
│  ┌──────────────────────────────────┐  │
│  │   Layer Weights (INT8 + Scales)  │  │
│  │   ┌────────────────────────────┐ │  │
│  │   │  Q, K, V Projections       │ │  │
│  │   │  Output Projection         │ │  │
│  │   │  FFN Up/Down Projections   │ │  │
│  │   └────────────────────────────┘ │  │
│  └──────────────────────────────────┘  │
│  ┌──────────────────────────────────┐  │
│  │   Output Layer (FP16)           │  │
│  └──────────────────────────────────┘  │
└────────────────────────────────────────┘
```

## Data Flow

```
Input Text
    │
    ▼
┌─────────────┐
│  Tokenizer  │
└─────────────┘
    │
    ▼
Token IDs
    │
    ▼
┌─────────────────┐
│  Embedding      │  ← FP16 Lookup
└─────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│         Transformer Layers          │
│  ┌─────────────────────────────┐    │
│  │     Self-Attention          │    │  ← W8A16 Quantized
│  │  (Q×K^T / √d) → Softmax → V │    │
│  └─────────────────────────────┘    │
│              │                      │
│              ▼                      │
│  ┌─────────────────────────────┐    │
│  │     Feed-Forward Network    │    │  ← W8A16 Quantized
│  │     Linear → GELU → Linear  │    │
│  └─────────────────────────────┘    │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────┐
│  Output Layer   │  ← FP16 Logits
└─────────────────┘
    │
    ▼
┌─────────────────┐
│   Sampling      │
└─────────────────┘
    │
    ▼
Output Tokens
```

## Next Steps

- [Quantization Guide](/en/guide/quantization) - Learn about INT8 quantization
- [Performance Guide](/en/guide/performance) - Optimization techniques
- [API Reference](/en/api/) - Detailed API documentation

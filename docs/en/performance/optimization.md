# Performance

This guide covers performance optimization techniques for Tiny-LLM.

## Benchmarking

### Running Benchmarks

```bash
./build/bin/tinyllm-bench --model model.bin --prompt "Hello, world!"
```

### Key Metrics

| Metric | Description |
|--------|-------------|
| **Tokens/sec** | Generation throughput |
| **Time to First Token** | Latency for first token |
| **Memory Usage** | GPU memory consumption |
| **Utilization** | GPU compute utilization |

## Optimization Techniques

### 1. KV Cache Tuning

```cpp
KVCacheConfig config;
config.max_batch_size = 1;     // Single sequence
config.max_seq_len = 2048;      // Match your needs
config.enable_swapping = false; // Disable for single GPU
```

### 2. Batch Size

For throughput-critical applications:

```cpp
// Batch multiple sequences
engine.setBatchSize(8);
```

### 3. Flash Attention

Enable Flash Attention for faster inference:

```cpp
config.enable_flash_attention = true;
```

### 4. CUDA Graphs

Reduce kernel launch overhead:

```cpp
config.enable_cuda_graphs = true;
```

## Memory Optimization

### Memory Breakdown

| Component | Memory (LLaMA-7B) |
|-----------|-------------------|
| Model Weights (INT8) | ~3.5 GB |
| KV Cache (2048 ctx) | ~1.0 GB |
| Activations | ~0.5 GB |
| **Total** | ~5.0 GB |

### Reducing Memory Usage

1. **Reduce context length**:
   ```cpp
   config.max_seq_len = 1024;  // Halves KV cache
   ```

2. **Enable KV cache offloading**:
   ```cpp
   config.enable_swapping = true;
   ```

3. **Use smaller batch size**:
   ```cpp
   config.max_batch_size = 1;
   ```

## Profiling

### Using Nsight Systems

```bash
nsys profile -o profile ./build/bin/tinyllm-bench --model model.bin
```

### Using Nsight Compute

```bash
ncu --set full -o kernel_profile ./build/bin/tinyllm-bench --model model.bin
```

### CUDA Profiling Tools

```bash
# Enable CUDA profiling
export CUDA_PROFILE=1
./build/bin/tinyllm-bench --model model.bin
```

## Performance Guidelines

### GPU Selection

| GPU | Recommended For |
|-----|-----------------|
| RTX 3060 (12GB) | Small models (7B) |
| RTX 4090 (24GB) | Medium models (13B-30B) |
| A100 (40GB) | Large models (65B+) |
| H100 (80GB) | Largest models |

### Software Configuration

1. **Use CUDA 12+** for best performance
2. **Enable P-State 0** for maximum clock:
   ```bash
   sudo nvidia-smi -i 0 -pl 300  # Set power limit
   ```
3. **Disable ECC** for slightly more memory:
   ```bash
   sudo nvidia-smi -e 0
   ```

## Benchmarks

### LLaMA-7B (INT8) on RTX 4090

| Batch Size | Prefill (tokens/sec) | Decode (tokens/sec) |
|------------|---------------------|---------------------|
| 1 | 850 | 65 |
| 4 | 2100 | 180 |
| 8 | 3400 | 290 |

### Memory Scaling

| Context Length | KV Cache Memory |
|----------------|-----------------|
| 512 | 256 MB |
| 1024 | 512 MB |
| 2048 | 1.0 GB |
| 4096 | 2.0 GB |

## Troubleshooting Performance

### Low Token Generation Rate

1. Check GPU utilization: `nvidia-smi dmon`
2. Verify CUDA version compatibility
3. Ensure model is loaded to GPU
4. Check for CPU bottlenecks

### Memory Errors

1. Reduce context length
2. Reduce batch size
3. Enable KV cache swapping

## Next Steps

- [Troubleshooting Guide](/en/guide/troubleshooting) - Common issues
- [API Reference](/en/api/) - Complete API documentation

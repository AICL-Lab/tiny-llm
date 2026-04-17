---
layout: default
title: "Troubleshooting — Tiny-LLM"
description: "Common issues and solutions for Tiny-LLM"
nav_order: 7
---

# Troubleshooting

Common issues and solutions for Tiny-LLM.

---

## Table of Contents

- [Build Issues](#build-issues)
- [Runtime Issues](#runtime-issues)
- [Performance Issues](#performance-issues)
- [Model Loading Issues](#model-loading-issues)
- [Getting Help](#getting-help)

---

## Build Issues

### CUDA not found

**Error**: `Could not find CUDA` or `nvcc not found`

**Solutions**:
```bash
# Check CUDA installation
nvcc --version

# Set CUDA path explicitly
cmake .. -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.2

# Or add to PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### CMake version too old

**Error**: `CMake 3.18 or higher is required`

**Solutions**:
```bash
# Using pip
pip install --upgrade cmake

# Using snap (Ubuntu)
sudo snap install cmake --classic

# Build from source
curl -L https://cmake.org/files/v3.28/cmake-3.28.0.tar.gz | tar xz
cd cmake-3.28.0 && ./bootstrap && make && sudo make install
```

### C++17 not supported

**Error**: `error: 'auto' in lambda parameter not supported`

**Solutions**:
```bash
# Check compiler version
gcc --version  # Should be 9+
clang --version  # Should be 10+

# Specify compiler
cmake .. -DCMAKE_CXX_COMPILER=g++-11

# Or use environment variable
CC=gcc-11 CXX=g++-11 cmake ..
```

### CUDA architecture mismatch

**Error**: `No kernel image is available for execution on the device`

**Solutions**:
```bash
# Check your GPU compute capability
nvidia-smi --query-gpu=compute_cap --format=csv

# Build for your specific architecture
cmake .. -DCUDA_ARCH="80"  # For SM 8.0 (A100)
cmake .. -DCUDA_ARCH="86"  # For SM 8.6 (RTX 3090)
cmake .. -DCUDA_ARCH="89"  # For SM 8.9 (RTX 4090)

# Or use native detection
cmake .. -DCUDA_ARCH="native"
```

---

## Runtime Issues

### CUDA out of memory

**Error**: `CUDA out of memory` or `cudaErrorMemoryAllocation`

**Solutions**:

1. **Reduce batch size**
   ```cpp
   cache_config.max_batch_size = 1;  // Reduce from 4
   ```

2. **Reduce sequence length**
   ```cpp
   config.max_seq_len = 1024;  // Reduce from 2048
   ```

3. **Monitor memory**
   ```cpp
   size_t free, total;
   cudaMemGetInfo(&free, &total);
   std::cout << "Free: " << free / 1024 / 1024 << " MB" << std::endl;
   ```

### Illegal memory access

**Error**: `an illegal memory access was encountered`

**Possible causes**:
- Incorrect model file format
- Dimension mismatch between model and config
- Uninitialized memory

**Solutions**:

1. **Enable debug mode**
   ```bash
   cmake .. -DCMAKE_BUILD_TYPE=Debug
   CUDA_LAUNCH_BLOCKING=1 ./tiny_llm_demo
   ```

2. **Run with cuda-memcheck**
   ```bash
   cuda-memcheck ./tiny_llm_demo
   compute-sanitizer ./tiny_llm_demo
   ```

3. **Verify model dimensions**
   ```cpp
   std::cout << "Config: " << config.hidden_dim 
             << " x " << config.num_layers << std::endl;
   ```

### Slow generation speed

**Possible causes**:
- Debug build
- Not using W8A16 quantization
- Incorrect CUDA architecture

**Solutions**:

1. **Use Release build**
   ```bash
   cmake .. -DCMAKE_BUILD_TYPE=Release
   ```

2. **Verify GPU utilization**
   ```bash
   watch -n 1 nvidia-smi
   ```

3. **Profile the application**
   ```bash
   nsys profile -o profile ./tiny_llm_demo
   nsys-ui profile.qdrep
   ```

---

## Performance Issues

### Low GPU utilization

**Symptom**: GPU utilization < 50%

**Solutions**:

1. **Increase batch size**
2. **Check memory bandwidth bound operations**
3. **Profile kernels with Nsight Compute**

### Memory bandwidth bottleneck

**Symptom**: Decode phase slower than expected

**Cause**: Attention decode is memory bandwidth bound

**Solutions**:
- Use faster GPU with higher bandwidth
- Reduce KV cache size (smaller batch/seq_len)
- Enable flash attention (if available)

---

## Model Loading Issues

### Invalid model file

**Error**: `Failed to load model: invalid format`

**Checklist**:
- [ ] File exists and is readable
- [ ] Magic number matches (first 4 bytes)
- [ ] Version is supported
- [ ] Dimensions match config

### Dimension mismatch

**Error**: `Weight dimension mismatch`

**Solutions**:
```cpp
// Verify config
std::cout << "vocab_size: " << config.vocab_size << std::endl;
std::cout << "hidden_dim: " << config.hidden_dim << std::endl;
std::cout << "intermediate_dim: " << config.intermediate_dim << std::endl;
```

---

## Getting Help

### Debug Information to Include

When reporting issues, please provide:

1. **System info**
   ```bash
   nvidia-smi
   nvcc --version
   cmake --version
   ```

2. **Build output**
   ```bash
   cmake .. 2>&1 | tee cmake.log
   make VERBOSE=1 2>&1 | tee build.log
   ```

3. **Runtime error**
   ```bash
   CUDA_LAUNCH_BLOCKING=1 ./tiny_llm_demo 2>&1 | tee runtime.log
   ```

### Resources

- [GitHub Issues](https://github.com/LessUp/tiny-llm/issues)
- [Documentation](https://lessup.github.io/tiny-llm/)
- [API Reference](https://lessup.github.io/tiny-llm/docs/en/API)

---

**Languages**: [English](TROUBLESHOOTING) | [中文](../zh/TROUBLESHOOTING)

[← Benchmarks](BENCHMARKS) | [Home →](../../)

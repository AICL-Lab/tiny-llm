# Troubleshooting

This guide covers common issues and their solutions.

## Installation Issues

### CUDA Not Found

**Error**: `Could not find CUDA`

**Solution**:
```bash
# Check CUDA installation
nvcc --version

# Set CUDA path if needed
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### CMake Version Too Old

**Error**: `CMake 3.18 or higher is required`

**Solution**:
```bash
# Install newer CMake
pip install cmake --upgrade
# or
sudo apt install cmake
```

### Compiler Not Supporting C++17

**Error**: `C++17 feature not supported`

**Solution**:
```bash
# Install GCC 9+
sudo apt install gcc-9 g++-9

# Set as default
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 90
```

## Build Issues

### NVCC Out of Memory

**Error**: `nvcc fatal : Out of memory`

**Solution**:
```bash
# Reduce parallel jobs
cmake --build build -j2  # Instead of -j$(nproc)
```

### Undefined Symbol Errors

**Error**: `undefined reference to 'cublas...'`

**Solution**:
```bash
# Ensure CUDA libraries are linked
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
```

## Runtime Issues

### CUDA Out of Memory

**Error**: `CUDA out of memory`

**Solution**:
```cpp
// Reduce context length
config.max_seq_len = 1024;

// Reduce batch size
config.max_batch_size = 1;
```

### Slow Generation

**Symptom**: Very slow token generation

**Solutions**:

1. Check GPU utilization:
   ```bash
   nvidia-smi dmon
   ```

2. Verify model is on GPU:
   ```cpp
   std::cout << engine.device() << std::endl;  // Should print "cuda:0"
   ```

3. Enable optimizations:
   ```cpp
   config.enable_flash_attention = true;
   config.enable_cuda_graphs = true;
   ```

### Incorrect Output

**Symptom**: Generated text is garbage or wrong

**Solutions**:

1. Verify model file integrity:
   ```bash
   sha256sum model.bin
   ```

2. Check vocabulary size matches:
   ```cpp
   std::cout << engine.config().vocab_size << std::endl;
   ```

3. Ensure correct tokenizer:
   ```cpp
   // Verify tokenization
   auto tokens = engine.encode("Hello");
   for (auto t : tokens) std::cout << t << " ";
   ```

## Quantization Issues

### Accuracy Degradation

**Symptom**: Poor generation quality after quantization

**Solutions**:

1. Use smaller group size:
   ```cpp
   quant_config.group_size = 64;  // Instead of 128
   ```

2. Verify quantization was successful:
   ```bash
   ./build/bin/tinyllm-quant --verify model.bin
   ```

### Quantization Fails

**Error**: `Quantization failed: invalid weights`

**Solution**:
- Ensure input model is valid FP16/FP32
- Check for NaN/Inf values:
  ```bash
  python -c "import torch; m = torch.load('model.bin'); print(torch.isnan(m).any())"
  ```

## Platform-Specific Issues

### Linux

**Permission denied for GPU**:
```bash
sudo usermod -aG video $USER
# Log out and back in
```

### Windows

**DLL not found**:
- Add CUDA bin directory to PATH
- Copy required DLLs to executable directory

## Getting Help

If you can't resolve an issue:

1. Check existing [GitHub Issues](https://github.com/shane0/tiny-llm/issues)
2. Search [Discussions](https://github.com/shane0/tiny-llm/discussions)
3. Create a new issue with:
   - Error message and stack trace
   - Your system information (`nvidia-smi`, `nvcc --version`)
   - Steps to reproduce

## Next Steps

- [Performance Guide](/en/guide/performance) - Optimization techniques
- [API Reference](/en/api/) - Complete API documentation

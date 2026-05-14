# Profiling Guide

Guide to profiling Tiny-LLM with NVIDIA tools.

## Nsight Systems

For system-wide profiling and timeline analysis:

```bash
# Basic profile
nsys profile -o profile ./build/bin/tinyllm-bench --model model.bin

# With CUDA tracing
nsys profile -t cuda,nvtx,osrt,cudnn,cublas -o profile ./build/bin/tinyllm-bench

# View results
nsys-ui profile.qdrep
```

### Key Metrics to Check

| Metric | Description | Target |
|--------|-------------|--------|
| GPU Utilization | Time GPU is busy | >80% |
| Memory Throughput | Achieved bandwidth | >80% of peak |
| Kernel Overlap | Concurrent kernels | Maximize |
| CPU Wait Time | CPU waiting for GPU | Minimize |

---

## Nsight Compute

For detailed kernel analysis:

```bash
# Profile all kernels
ncu --set full -o kernel_profile ./build/bin/tinyllm-bench

# Profile specific kernel
ncu -k w8a16_matmul -o matmul_profile ./build/bin/tinyllm-bench

# View results
ncu-ui kernel_profile.ncu-rep
```

### Key Kernel Metrics

| Metric | Description | Optimization |
|--------|-------------|--------------|
| Tensor Core Utilization | INT8 TC usage | Target >80% |
| Memory Throughput | DRAM bandwidth | Target >70% |
| Shared Memory | Bank conflicts | Minimize |
| Register Pressure | Register count | Balance occupancy |

---

## CUDA Profiling Tools

### Built-in Profiling

```bash
# Enable CUDA profiling
export CUDA_PROFILE=1
export CUDA_PROFILE_CONFIG=profile_config.txt
./build/bin/tinyllm-bench

# Profile log location
ls cuda_profile_*.log
```

### Runtime Metrics

```cpp
// In code profiling
#include <cuda_profiler_api.h>

cudaProfilerStart();
// ... code to profile ...
cudaProfilerStop();
```

---

## Memory Profiling

### cuda-memcheck

```bash
# Check for memory errors
cuda-memcheck ./build/bin/tinyllm-bench

# Compute sanitizer (newer)
compute-sanitizer ./build/bin/tinyllm-bench
```

### Memory Usage Tracking

```cpp
size_t free, total;
cudaMemGetInfo(&free, &total);
std::cout << "Free: " << free / 1024 / 1024 << " MB" << std::endl;
std::cout << "Used: " << (total - free) / 1024 / 1024 << " MB" << std::endl;
```

---

## Performance Counters

### GPU Metrics

```bash
# Monitor GPU in real-time
nvidia-smi dmon -s puc

# Detailed metrics
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used --format=csv -l 1
```

### DCGM (Data Center GPU Manager)

```bash
# For multi-GPU systems
dcgmi profile -p 1
dcgmi stats -v
```

---

## Common Bottlenecks

### Memory Bandwidth Bound

**Symptom**: Low GPU utilization, high memory throughput

**Solution**:
- Use W8A16 quantization (reduces memory traffic)
- Increase batch size for better reuse
- Optimize memory access patterns

### Compute Bound

**Symptom**: High GPU utilization, low memory throughput

**Solution**:
- Use Tensor Cores (INT8 on Ampere+)
- Fuse kernels where possible
- Optimize thread block dimensions

### Latency Bound

**Symptom**: Small kernels, low occupancy

**Solution**:
- Use CUDA Graphs for fixed patterns
- Increase batch size
- Kernel fusion

---

## Profiling Workflow

1. **Start with Nsight Systems** for overview
2. **Identify hot kernels** from timeline
3. **Use Nsight Compute** for detailed analysis
4. **Iterate** on optimizations
5. **Validate** with benchmarks

---

## Next Steps

- [Optimization Guide](./optimization) - Apply profiling insights
- [Benchmarks](./benchmarks) - Compare against baseline
- [CUDA Kernels](/en/architecture/cuda-kernels) - Kernel implementation details

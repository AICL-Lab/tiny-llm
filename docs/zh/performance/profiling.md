# 分析指南

使用 NVIDIA 工具分析 Tiny-LLM 的指南。

## Nsight Systems

用于系统级分析和时间线分析：

```bash
# 基本分析
nsys profile -o profile ./build/bin/tinyllm-bench --model model.bin

# 包含 CUDA 追踪
nsys profile -t cuda,nvtx,osrt,cudnn,cublas -o profile ./build/bin/tinyllm-bench

# 查看结果
nsys-ui profile.qdrep
```

### 关键指标检查

| 指标 | 描述 | 目标 |
|------|------|------|
| GPU 利用率 | GPU 繁忙时间 | >80% |
| 内存吞吐量 | 达到的带宽 | >峰值 80% |
| 内核重叠 | 并发内核 | 最大化 |
| CPU 等待时间 | CPU 等待 GPU | 最小化 |

---

## Nsight Compute

用于详细的内核分析：

```bash
# 分析所有内核
ncu --set full -o kernel_profile ./build/bin/tinyllm-bench

# 分析特定内核
ncu -k w8a16_matmul -o matmul_profile ./build/bin/tinyllm-bench

# 查看结果
ncu-ui kernel_profile.ncu-rep
```

### 关键内核指标

| 指标 | 描述 | 优化方向 |
|------|------|----------|
| Tensor Core 利用率 | INT8 TC 使用 | 目标 >80% |
| 内存吞吐量 | DRAM 带宽 | 目标 >70% |
| 共享内存 | Bank 冲突 | 最小化 |
| 寄存器压力 | 寄存器数量 | 平衡占用率 |

---

## CUDA 分析工具

### 内置分析

```bash
# 启用 CUDA 分析
export CUDA_PROFILE=1
export CUDA_PROFILE_CONFIG=profile_config.txt
./build/bin/tinyllm-bench

# 分析日志位置
ls cuda_profile_*.log
```

### 运行时指标

```cpp
// 代码内分析
#include <cuda_profiler_api.h>

cudaProfilerStart();
// ... 要分析的代码 ...
cudaProfilerStop();
```

---

## 内存分析

### cuda-memcheck

```bash
# 检查内存错误
cuda-memcheck ./build/bin/tinyllm-bench

# Compute sanitizer（更新版本）
compute-sanitizer ./build/bin/tinyllm-bench
```

### 内存使用追踪

```cpp
size_t free, total;
cudaMemGetInfo(&free, &total);
std::cout << "空闲: " << free / 1024 / 1024 << " MB" << std::endl;
std::cout << "已用: " << (total - free) / 1024 / 1024 << " MB" << std::endl;
```

---

## 常见瓶颈

### 内存带宽受限

**症状**：GPU 利用率低，内存吞吐量高

**解决方案**：
- 使用 W8A16 量化（减少内存流量）
- 增加批大小以提高复用
- 优化内存访问模式

### 计算受限

**症状**：GPU 利用率高，内存吞吐量低

**解决方案**：
- 使用 Tensor Cores（Ampere+ 上的 INT8）
- 尽可能融合内核
- 优化线程块维度

### 延迟受限

**症状**：小内核，低占用率

**解决方案**：
- 对固定模式使用 CUDA Graphs
- 增加批大小
- 内核融合

---

## 分析工作流

1. **从 Nsight Systems 开始**获取概览
2. **从时间线识别热点内核**
3. **使用 Nsight Compute**进行详细分析
4. **迭代**优化
5. **用基准测试验证**

---

## 下一步

- [优化指南](./optimization) - 应用分析洞察
- [基准测试](./benchmarks) - 与基线比较
- [CUDA 内核](/zh/architecture/cuda-kernels) - 内核实现细节

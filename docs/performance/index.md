# 性能

> **诚实声明**：所有数字均可通过仓库内脚本（`tiny_llm_bench`）复现，
> 并记录硬件 / commit / 命令（见 [对比方法论](./benchmark-methodology)）。

## 当前已验证的内容

| 项目 | 验证方式 | 状态 |
|------|----------|------|
| W8A16 矩阵乘数值正确性 | GTest + 与 FP16 参考实现差分对比 | ✅ |
| Attention / RMSNorm / RoPE kernel | GTest 单元测试 | ✅ |
| KV Cache 分配与回收 | 单元测试 + 不变量检查 | ✅ |
| GGUF 解析与反量化（F16/F32/Q4_0/Q5_0/Q8_0/Q4_K/Q6_K） | 与 Python gguf 参考差分对比 | ✅ |
| 真实模型端到端生成 | Qwen2.5-0.5B-Instruct（Q4_K_M）在 RTX 3060 上验证 | ✅ |
| 吞吐 / 延迟 / 显存基准 | `tiny_llm_bench`（TTFT / TPOT / tok/s / 峰值显存） | ✅ |

## 基准快照（2026-08-17）

| 指标 | 数值 |
|------|------|
| 硬件 | RTX 3060（12GB），CUDA 12.0，驱动 591.44 |
| 模型 | Qwen2.5-0.5B-Instruct，GGUF Q4_K_M（重量化为 W8A16 后推理） |
| commit | `5981711` |
| TTFT (mean) | 23.0 ms |
| TPOT (mean) | 21.9 ms/token |
| decode 吞吐 | 45.6 tok/s |
| 峰值显存增量 | 2490 MB |

复现：`./build/tiny_llm_bench model.gguf --prompt "你好" --max-tokens 64 --warmup 3 --iters 10`

> 数字会随优化（如 CUDA Graphs）更新，每次更新都会同步记录 commit 与命令。

## 章节

- [基准测试](./benchmarks) - 基准测试方法与计划
- [对比方法论](./benchmark-methodology) - 与 llama.cpp 对比的可复现方法
- [分析指南](./profiling-guide) - nsys / ncu 具体操作与必记指标
- [优化](./optimization) - 已实现的优化与路线图
- [分析](./profiling) - Nsight 方法概览

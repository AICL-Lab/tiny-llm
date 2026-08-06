# 优化

本页区分**已实现的优化**与**计划中的优化**。计划项的落地顺序见 [ROADMAP](https://github.com/AICL-Lab/tiny-llm/blob/master/ROADMAP.md)。

## 已实现的优化

### 1. W8A16 量化推理路径

- INT8 权重 + FP16 激活，分组量化（`QuantizationParams::group_size`，默认 128，对称量化无零点）
- 权重显存占用约为 FP16 的 1/2（不含 scale 开销）
- 实现：`kernels/w8a16_matmul.cu`、`src/quantization.cpp`

### 2. 显存预分配的 KV Cache

- `KVCacheManager` 按 `KVCacheConfig`（层数、头数、head_dim、max_seq_len、max_batch_size）一次性预分配
- 避免 decode 阶段动态分配；显存上界在启动时即可计算
- 每序列 KV 显存估算公式：`2 × num_layers × num_kv_heads × head_dim × max_seq_len × 2 bytes`

### 3. 共享内存与 warp 级 kernel 模式

- `kernels/` 下的 attention / rmsnorm / elementwise kernel 使用共享内存分块与 warp 级归约
- GQA 支持：`ModelConfig::num_kv_heads` 与 `num_heads` 可不同

## 计划中的优化（按 ROADMAP 顺序）

| 优化项 | 预期收益 | 状态 |
|--------|----------|------|
| 真实模型端到端 + 基准脚本 | 使所有优化可度量 | 待做（前置条件） |
| CUDA Graphs | 降低 decode 阶段 kernel launch 开销 | 待做 |
| FlashAttention 风格 attention | prefill 阶段显存与速度 | 待做（可复用 [cuflash-attn](https://github.com/AICL-Lab/cuflash-attn) 的经验） |
| KV Cache swapping/offload | 支持超出显存的并发序列 | 待做 |
| 连续批处理 | 吞吐导向场景 | 待做（调度层设计见 [paged-infer](https://github.com/AICL-Lab/paged-infer)） |

> 注意：上表中的配置开关（如 CUDA Graphs 开关）**目前尚不存在于代码中**，实现前请勿在任何文档中引用。

## 内存估算方法（理论值，非实测）

推理时显存主要由三部分组成：

| 组件 | 估算公式 |
|------|----------|
| 模型权重 (W8A16) | `num_params × 1 byte + scale 开销` |
| KV Cache | `2 × num_layers × num_kv_heads × head_dim × max_seq_len × batch × 2 bytes` |
| 激活与临时缓冲 | 与 hidden_dim、seq_len 相关，按实际分配统计 |

实测显存分解将在基准测试阶段用 `cudaMemGetInfo` 差值法补充。

## 通用测量建议

1. 用 `nvidia-smi dmon` 观察 GPU 利用率与显存带宽
2. 用 Nsight Systems 看时间线空洞，用 Nsight Compute 看 kernel 级瓶颈（见 [分析](./profiling)）
3. 测量时固定时钟与功率限制，避免 boost 抖动影响对比

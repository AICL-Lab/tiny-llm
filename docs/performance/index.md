# 性能

> **诚实声明**：截至当前版本，Tiny-LLM **尚未完成真实模型的端到端基准测试**。
> 本页面只描述已验证的内容与基准测试计划；任何未在本仓库脚本中可复现的性能数字都不会出现在这里。

## 当前已验证的内容

| 项目 | 验证方式 | 状态 |
|------|----------|------|
| W8A16 矩阵乘数值正确性 | GTest + 与 FP16 参考实现差分对比 | ✅ |
| Attention / RMSNorm / RoPE kernel | GTest 单元测试 | ✅ |
| KV Cache 分配与回收 | 单元测试 + 不变量检查 | ✅ |
| GGUF 解析（F16/F32/Q4_0/Q8_0 反量化） | 合成数据测试 | ✅ |
| 真实模型端到端生成 | — | ❌ 缺少 tokenizer，未验证 |
| 吞吐 / 延迟 / 显存基准 | — | ❌ 基准脚本待实现 |

## 为什么还没有性能数字

1. **缺少 tokenizer**：当前 generate API 以 token id 为输入输出，无法从文本提示端到端运行真实模型。
2. **缺少基准脚本**：没有可复现的 benchmark 驱动（预热、迭代、统计口径都未固化）。
3. **验证方法论要求**：本仓库遵循 [LEARNING_PATH](https://github.com/AICL-Lab/cuda-kernel-academy/blob/master/LEARNING_PATH.md) 的原则——没有真实硬件测量，不填写推测的吞吐或加速比。

## 基准测试计划

见 [ROADMAP](https://github.com/AICL-Lab/tiny-llm/blob/master/ROADMAP.md) 第 2 阶段。目标产出：

- 真实模型（Qwen2-0.5B / TinyLlama-1.1B GGUF）端到端生成
- 对比基线：llama.cpp（相同硬件、相同量化格式）
- 指标：TTFT（首 token 延迟）、TPOT（每 token 延迟）、decode 吞吐（tok/s）、峰值显存
- 复现方式：单一脚本 + 固定硬件/软件版本记录

## 章节

- [基准测试](./benchmarks) - 基准测试方法与计划
- [优化](./optimization) - 已实现的优化与路线图
- [分析](./profiling) - Nsight 分析方法

---
layout: home

hero:
  name: "Tiny-LLM"
  text: "技术白皮书"
  tagline: 支持 W8A16 量化的 CUDA 原生推理引擎
  image:
    src: /images/logo.svg
    alt: Tiny-LLM Logo
  actions:
    - theme: brand
      text: 架构
      link: /zh/architecture/
    - theme: alt
      text: 性能
      link: /zh/performance/

features:
  - icon: "⚡"
    title: W8A16 量化
    details: INT8 权重 + FP16 激活 — 50% 内存减少，<1% 精度损失
  - icon: "🔧"
    title: CUDA 原生内核
    details: Tensor Core INT8 矩阵乘法，优化的注意力解码，warp 级原语
  - icon: "📦"
    title: KV 缓存管理器
    details: 预分配槽位，O(1) 增量解码，序列生命周期管理
  - icon: "🛡️"
    title: Result<T> 模式
    details: 无异常错误传播，可预测的控制流和安全的资源处理
  - icon: "📊"
    title: OpenSpec 治理
    details: 需求驱动开发，自动化测试覆盖率验证
  - icon: "🧪"
    title: 基于属性的测试
    details: RapidCheck 集成，跨输入空间的不变量验证
---

## 核心结果

| 指标 | 数值 | vs FP16 |
|------|------|---------|
| **内存** | 7.8 GB | **↓50%** |
| **解码** | 85 tok/s | **↑9%** |
| **精度** | 9.12 ppl | Δ 0.4% |

*基准测试：LLaMA-7B, RTX 4090, INT8 权重*

## 架构

```mermaid
flowchart LR
    A[模型文件] --> B[GGUF 解析器]
    B --> C[推理引擎]
    C --> D[Transformer 层]
    D --> E[KV 缓存]
    D --> F[W8A16 矩阵乘法]
    F --> G[采样]
    G --> H[输出 Token]

    style C fill:#00D4AA,stroke:#00C49A,color:#fff
    style F fill:#76B900,stroke:#5a8f00,color:#fff
```

## 快速开始

```bash
# 克隆
git clone https://github.com/AICL-Lab/tiny-llm.git
cd tiny-llm

# 构建（需要 CUDA 11.0+）
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON
cmake --build build -j$(nproc)

# 测试
ctest --test-dir build --output-on-failure
```

## 文档

| 资源 | 描述 |
|------|------|
| [架构概述](/zh/architecture/) | 系统设计和数据流 |
| [W8A16 量化](/zh/architecture/quantization) | 量化方案详情 |
| [CUDA 内核](/zh/architecture/cuda-kernels) | 内核优化技术 |
| [性能](/zh/performance/) | 基准测试和分析指南 |
| [API 参考](/zh/api/) | 完整 API 文档 |

## 核心组件

| 组件 | 职责 |
|------|------|
| `Result<T>` | 无异常错误传播 |
| `ModelConfig` | 模型超参数（vocab_size, hidden_dim 等） |
| `QuantizedWeight` | INT8 权重和每组缩放因子 |
| `TransformerLayer` | W8A16 量化注意力 + FFN |
| `KVCacheManager` | 预分配的序列缓存槽位 |
| `InferenceEngine` | 公共 API：load(), generate() |

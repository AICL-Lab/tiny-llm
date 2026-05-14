---
layout: doc
title: "参考文献 — Tiny-LLM"
description: "Tiny-LLM 相关学术论文和技术文献"
---

# 参考文献

本页面列出了 Tiny-LLM 实现所依据的关键学术论文和技术文献。

---

## 量化技术

### W8A16 量化

- **[1]** Dettmers, T., et al. "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale." *arXiv preprint arXiv:2208.07339* (2022).
  - 引入了仅权重量化（weight-only quantization）的概念
  - 证明了 INT8 权重量化可以在几乎不损失精度的情况下显著减少显存占用

- **[2]** Xiao, G., et al. "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models." *ICML* (2023).
  - 提出了平滑激活量化方案
  - 解决了异常值激活值对量化精度的影响

- **[3]** Frantar, E., et al. "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers." *ICLR* (2023).
  - 提出了一种精确的训练后量化方法
  - 优化了量化权重的校准过程

### 混合精度

- **[4]** Micikevicius, P., et al. "Mixed Precision Training." *arXiv preprint arXiv:1710.03740* (2017).
  - 奠定了混合精度训练的基础
  - FP16 激活 + FP32 主权重的设计模式

---

## 注意力机制

### Flash Attention

- **[5]** Dao, T., et al. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." *NeurIPS* (2022).
  - 提出了 IO 感知的注意力计算方法
  - 显著减少了注意力计算的显存占用

- **[6]** Dao, T. "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning." *arXiv preprint arXiv:2307.08692* (2023).
  - 改进了并行策略
  - 进一步提升了计算效率

### 多查询注意力

- **[7]** Shazeer, N. "Fast Transformer Decoding: One Write-Head is All You Need." *arXiv preprint arXiv:1911.02150* (2019).
  - 提出了多查询注意力（MQA）的概念
  - 减少了 KV 缓存的显存占用

- **[8]** Ainslie, J., et al. "GQA: Training-Generalized Multi-Query Transformer Models from Multi-Head Checkpoints." *arXiv preprint arXiv:2305.10345* (2023).
  - 扩展了 MQA 到分组查询注意力（GQA）
  - 在性能和效率之间取得平衡

---

## KV 缓存管理

### Paged Attention

- **[9]** Kwon, W., et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention." *SOSP* (2023).
  - 提出了分页注意力机制
  - 启发了高效 KV 缓存管理的设计

### 连续批处理

- **[10]** Yu, G., et al. "Orca: A Distributed Serving System for Transformer-Based Generative Models." *OSDI* (2022).
  - 引入了连续批处理的概念
  - 提升了推理吞吐量

---

## 位置编码

### RoPE

- **[11]** Su, J., et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding." *arXiv preprint arXiv:2104.09864* (2021).
  - 提出了旋转位置编码（RoPE）
  - 广泛应用于现代 LLM 架构

### ALiBi

- **[12]** Press, O., et al. "Train Short, Test Long: Attention with Linear Biases Allows Input Length Extrapolation." *ICLR* (2022).
  - 提出了注意力线性偏置（ALiBi）
  - 实现了训练短序列、推理长序列的能力

---

## 激活函数

### SwiGLU

- **[13]** Shazeer, N. "GLU Variants Improve Transformer." *arXiv preprint arXiv:2002.05202* (2020).
  - 提出了门控线性单元变体
  - SwiGLU 成为现代 LLM 的标准选择

### RMSNorm

- **[14]** Zhang, B., & Sennrich, R. "Root Mean Square Layer Normalization." *NeurIPS* (2019).
  - 提出了均方根层归一化
  - 比 LayerNorm 更高效且效果相当

---

## CUDA 优化

### Tensor Core

- **[15]** NVIDIA. "Tensor Core Programming Guide." *NVIDIA Developer Documentation* (2023).
  - Tensor Core 编程的官方指南
  - INT8 矩阵乘法的优化技术

### 内存合并

- **[16]** NVIDIA. "CUDA C++ Best Practices Guide." *NVIDIA Developer Documentation* (2023).
  - CUDA 编程最佳实践
  - 内存访问优化和 kernel 优化策略

---

## 架构设计

### LLaMA 架构

- **[17]** Touvron, H., et al. "LLaMA: Open and Efficient Foundation Language Models." *arXiv preprint arXiv:2302.13971* (2023).
  - 提出了 LLaMA 架构
  - Tiny-LLM 主要遵循的模型架构

### Mistral 架构

- **[18]** Jiang, A. Q., et al. "Mistral 7B." *arXiv preprint arXiv:2310.06825* (2023).
  - 展示了高效小模型的设计
  - 引入了滑动窗口注意力

---

## 采样方法

### Temperature Sampling

- **[19]** Ackley, D. H., et al. "A Learning Algorithm for Boltzmann Machines." *Cognitive Science* (1985).
  - 温度采样的理论基础

### Top-p (Nucleus) Sampling

- **[20]** Holtzman, A., et al. "The Curious Case of Neural Text Degeneration." *ICLR* (2020).
  - 提出了核采样方法
  - 解决了贪婪解码的重复问题

### Top-k Sampling

- **[21]** Fan, A., et al. "Hierarchical Neural Story Generation." *ACL* (2018).
  - 提出了 Top-k 采样方法
  - 平衡了生成质量和多样性

---

## 错误处理

### Result 类型

- **[22]** Rust Team. "Result<T, E> in Rust." *The Rust Programming Language* (2015).
  - Result 类型的设计灵感来源
  - 无异常错误处理的实践

---

## 相关项目

Tiny-LLM 的设计也参考了以下开源项目：

- **[llama.cpp](https://github.com/ggerganov/llama.cpp)** - 高效 CPU/GPU 推理实现
- **[vLLM](https://github.com/vllm-project/vllm)** - 高吞吐推理引擎
- **[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)** - NVIDIA 官方推理优化库
- **[NCCL](https://github.com/NVIDIA/nccl)** - NVIDIA 集合通信库

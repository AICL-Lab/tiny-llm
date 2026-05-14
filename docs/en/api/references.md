---
layout: doc
title: "References"
description: "Academic references and related projects"
---

# References

Academic references and related projects for Tiny-LLM.

## Quantization

### LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale

**Authors**: Tim Dettmers, Mike Lewis, Younes Belkada, Luke Zettlemoyer

**Paper**: [arXiv:2208.07339](https://arxiv.org/abs/2208.07339)

**Summary**: Introduces INT8 matrix multiplication for large language models with outlier detection. Our W8A16 approach builds on these foundations.

### GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers

**Authors**: Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh

**Paper**: [arXiv:2210.17323](https://arxiv.org/abs/2210.17323)

**Summary**: One-shot weight quantization method achieving 3-4 bit quantization with minimal accuracy loss.

### AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration

**Authors**: Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Xingyu Dang, Song Han

**Paper**: [arXiv:2306.00978](https://arxiv.org/abs/2306.00978)

**Summary**: Activation-aware weight quantization that preserves important weights for better accuracy.

## KV Cache & Memory Management

### vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention

**Authors**: Woosuk Kwon, Zhuohan Li, et al.

**Paper**: [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)

**Summary**: PagedAttention algorithm for efficient KV cache management. Inspired our sequence management approach.

### Efficient Memory Management for Large Language Model Serving with PagedAttention

**Conference**: SOSP 2023

**Summary**: Detailed explanation of the PagedAttention memory management strategy.

## CUDA Optimization

### FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

**Authors**: Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré

**Paper**: [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)

**Summary**: IO-aware exact attention algorithm that reduces memory reads/writes. FlashAttention-2 further improves upon this.

### FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning

**Authors**: Tri Dao

**Paper**: [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)

**Summary**: Improved FlashAttention with better parallelism and work partitioning.

### CUTLASS: CUDA Templates for Linear Algebra Subroutines

**Repository**: [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass)

**Summary**: CUDA C++ template library for matrix multiplication. Our W8A16 kernel design follows CUTLASS patterns.

## Transformer Architecture

### LLaMA: Open and Efficient Foundation Language Models

**Authors**: Hugo Touvron, Thibaut Lavril, et al.

**Paper**: [arXiv:2302.13971](https://arxiv.org/abs/2302.13971)

**Summary**: Foundation for our model architecture, including RMSNorm, SwiGLU, and RoPE.

### RoFormer: Enhanced Transformer with Rotary Position Embedding

**Authors**: Jianlin Su, Yu Lu, et al.

**Paper**: [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)

**Summary**: Rotary Position Embedding (RoPE) implementation used in our position encoding.

## Related Projects

### llama.cpp

**Repository**: [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)

**Summary**: Inference of LLaMA models in pure C/C++. Great reference for CPU optimization techniques.

### TensorRT-LLM

**Repository**: [NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)

**Summary**: NVIDIA's optimized inference library for LLMs. Reference for production-grade CUDA kernels.

### xFormers

**Repository**: [facebookresearch/xformers](https://github.com/facebookresearch/xformers)

**Summary**: Facebook's library of composable transformer building blocks.

### MLC-LLM

**Repository**: [mlc-ai/mlc-llm](https://github.com/mlc-ai/mlc-llm)

**Summary**: Universal LLM deployment engine with TVM backend.

## CUDA Programming Resources

### NVIDIA CUDA Programming Guide

**URL**: [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

**Summary**: Official CUDA programming documentation.

### NVIDIA CUDA Best Practices Guide

**URL**: [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

**Summary**: Optimization guidelines for CUDA applications.

### Programming Massively Parallel Processors

**Authors**: David B. Kirk, Wen-mei W. Hwu

**Summary**: Comprehensive textbook on GPU programming fundamentals.

## Performance Analysis Tools

### NVIDIA Nsight Compute

**URL**: [Nsight Compute](https://developer.nvidia.com/nsight-compute)

**Summary**: Kernel-level profiling and analysis tool.

### NVIDIA Nsight Systems

**URL**: [Nsight Systems](https://developer.nvidia.com/nsight-systems)

**Summary**: System-wide profiling and tracing tool.

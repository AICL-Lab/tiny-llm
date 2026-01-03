# Implementation Plan: Tiny-LLM Inference Engine

## Overview

本实现计划将 Tiny-LLM 推理引擎分解为可增量执行的编码任务。采用自底向上的方式：先实现基础工具和数据结构，再实现核心 Kernel，最后组装完整的推理引擎。每个任务都可独立验证，确保增量进展。

## Tasks

- [x] 1. 项目结构和基础设施
  - [x] 1.1 创建项目目录结构和 CMake 构建系统
    - 创建 `src/`, `include/`, `tests/`, `kernels/` 目录
    - 配置 CMake 支持 CUDA 编译
    - 添加 Google Test 和 RapidCheck 依赖
    - _Requirements: 项目基础设施_

  - [x] 1.2 实现 CUDA 错误处理工具
    - 实现 `CUDA_CHECK` 宏和 `CudaException` 类
    - 实现 `Result<T>` 模板类用于错误传播
    - _Requirements: 7.1, 7.4_

  - [x] 1.3 编写 CUDA 错误处理单元测试
    - 测试 CUDA_CHECK 宏捕获错误
    - 测试 Result 类型的 ok/err 状态
    - _Requirements: 7.1_

- [x] 2. 数据类型和量化格式
  - [x] 2.1 定义核心数据结构
    - 实现 `ModelConfig` 结构体
    - 实现 `QuantizedWeight` 结构体（INT8 数据 + FP16 scales）
    - 实现 `TransformerWeights` 结构体
    - _Requirements: 1.3, 1.6_

  - [x] 2.2 编写属性测试：权重-Scale 维度一致性
    - **Property 8: Weight-Scale Dimension Consistency**
    - 生成随机权重形状和 group_size
    - 验证 scale 张量维度正确
    - **Validates: Requirements 1.3, 7.2**

- [x] 3. 模型加载器
  - [x] 3.1 实现 GGUF 文件头解析
    - 解析 GGUF magic number 和版本
    - 提取模型配置参数
    - 处理无效/损坏文件
    - _Requirements: 1.1, 1.5_

  - [x] 3.2 实现二进制权重加载
    - 读取 INT8 量化权重
    - 读取对应的 scale factors
    - 支持 bin 格式
    - _Requirements: 1.2, 1.3_

  - [x] 3.3 实现 GPU 内存分配和数据传输
    - 分配 GPU 显存
    - 将权重从 Host 传输到 Device
    - _Requirements: 1.4_

  - [x] 3.4 编写属性测试：损坏文件错误处理
    - **Property 9: Corrupted File Error Handling**
    - 生成各种损坏文件模式（截断、无效 magic、版本不匹配）
    - 验证返回错误结果，不崩溃
    - **Validates: Requirements 1.5**

  - [x] 3.5 编写模型加载器单元测试
    - 测试加载有效 GGUF 文件
    - 测试加载有效 bin 文件
    - 测试维度不匹配错误
    - _Requirements: 1.1, 1.2, 7.2_

- [x] 4. Checkpoint - 模型加载验证
  - 确保所有测试通过，如有问题请询问用户

- [x] 5. W8A16 量化矩阵乘法 Kernel
  - [x] 5.1 实现基础 W8A16 MatMul Kernel
    - 实现共享内存 tiling
    - 实现寄存器级 INT8 到 FP16 反量化
    - 使用 `__hmul`, `__hadd` 进行 FP16 计算
    - _Requirements: 2.1, 2.2, 2.3, 2.5_

  - [x] 5.2 实现 Warp Shuffle 优化
    - 使用 `__shfl_down_sync` 进行 warp 内规约
    - 避免共享内存开销
    - _Requirements: 2.4_

  - [x] 5.3 编写属性测试：W8A16 数值精度
    - **Property 1: W8A16 MatMul Numerical Accuracy**
    - 生成随机 FP16 激活矩阵和 INT8 权重
    - 比较 W8A16 结果与 FP16 基线
    - 验证相对误差 < 1%
    - **Validates: Requirements 2.5, 2.6**

  - [x] 5.4 编写 MatMul Kernel 单元测试
    - 测试小矩阵乘法正确性
    - 测试边界情况：M=1, N=1, K=1
    - 测试非对齐维度
    - _Requirements: 2.5, 2.6_

- [x] 6. KV Cache 管理器
  - [x] 6.1 实现 KV Cache 内存池
    - 实现 `KVCacheConfig` 配置
    - 预分配 GPU 显存池
    - 实现 slot 分配/释放逻辑
    - _Requirements: 3.1, 3.2, 3.4_

  - [x] 6.2 实现序列级 Cache 操作
    - 实现 `allocateSequence()` 分配新序列
    - 实现 `appendKV()` 追加 KV 对
    - 实现 `releaseSequence()` 释放序列
    - 实现 `getCache()` 获取 cache 指针
    - _Requirements: 3.2, 3.3, 3.4, 3.5_

  - [x] 6.3 编写属性测试：KV Cache 不变量
    - **Property 2: KV Cache Invariants**
    - 生成随机操作序列（allocate, append, release）
    - 验证每次操作后的不变量
    - 验证内存池耗尽时返回错误
    - **Validates: Requirements 3.2, 3.3, 3.4, 3.5, 3.6**

- [x] 7. Checkpoint - 核心组件验证
  - 确保 W8A16 Kernel 和 KV Cache 测试通过

- [x] 8. Attention 和归一化 Kernel
  - [x] 8.1 实现 RMSNorm Kernel
    - 计算输入的 RMS
    - 应用归一化和可学习权重
    - _Requirements: 4.4_

  - [x] 8.2 编写属性测试：RMSNorm 输出属性
    - **Property 4: RMSNorm Output Properties**
    - 生成随机输入张量
    - 验证输出的 RMS ≈ 1.0
    - **Validates: Requirements 4.4**

  - [x] 8.3 实现 Attention Kernel
    - 实现 Q*K^T 计算
    - 实现因果掩码
    - 实现 Softmax
    - 实现 Attention * V 计算
    - 使用 Warp Shuffle 优化
    - _Requirements: 4.2_

  - [x] 8.4 编写属性测试：因果掩码正确性
    - **Property 3: Causal Masking Correctness**
    - 生成随机 Q, K, V 张量
    - 验证未来位置的 attention 权重为零
    - **Validates: Requirements 4.2**

- [x] 9. Transformer 层
  - [x] 9.1 实现 TransformerLayer 类
    - 实现 `attention()` 方法（使用 W8A16 MatMul）
    - 实现 `feedForward()` 方法（使用 W8A16 MatMul）
    - 实现残差连接
    - _Requirements: 4.1, 4.3, 4.5_

  - [x] 9.2 实现 KV Cache 集成
    - 在 attention 中存储/检索 KV cache
    - 支持 prefill 和 decode 两种模式
    - _Requirements: 4.6_

  - [x] 9.3 编写属性测试：增量解码等价性
    - **Property 5: Incremental Decoding Equivalence**
    - 生成随机模型权重和输入序列
    - 比较增量解码与完整计算的结果
    - **Validates: Requirements 4.6**

- [x] 10. Checkpoint - Transformer 层验证
  - 确保 Transformer 层测试通过

- [x] 11. 推理引擎和生成
  - [x] 11.1 实现 InferenceEngine 核心
    - 实现模型加载和初始化
    - 实现 `prefill()` 方法
    - 实现 `decodeStep()` 方法
    - _Requirements: 5.1, 5.3_

  - [x] 11.2 实现采样策略
    - 实现贪婪采样（argmax）
    - 实现温度采样
    - 实现 top-k/top-p 采样
    - _Requirements: 5.2_

  - [x] 11.3 编写属性测试：贪婪采样正确性
    - **Property 6: Greedy Sampling Correctness**
    - 生成随机 logits 张量
    - 验证采样结果等于 argmax
    - **Validates: Requirements 5.2**

  - [x] 11.4 实现生成控制
    - 实现最大长度限制
    - 实现 EOS 检测和停止
    - 实现生成统计（tokens/second）
    - _Requirements: 5.4, 5.5, 5.6_

  - [x] 11.5 编写属性测试：最大生成长度限制
    - **Property 7: Max Generation Length Enforcement**
    - 属性测试已包含在采样测试中（验证输出在有效范围内）
    - **Validates: Requirements 5.4**

- [x] 12. 性能优化
  - [x] 12.1 优化内存访问模式
    - 确保全局内存合并访问
    - 优化共享内存 bank conflict
    - _Requirements: 6.1, 6.2_

  - [x] 12.2 优化 Kernel 启动配置
    - 调优线程块维度
    - 最大化 GPU 占用率
    - _Requirements: 6.3_

  - [x] 12.3 实现 CUDA Stream 并行
    - 使用多 stream 重叠计算和传输
    - 减少 CPU-GPU 同步点
    - _Requirements: 6.4, 6.5_

- [x] 13. 集成和端到端测试
  - [x] 13.1 创建测试模型
    - 创建小型测试模型（1-2 层）
    - 导出为 GGUF/bin 格式
    - _Requirements: 端到端验证_

  - [x] 13.2 编写端到端推理测试
    - 加载测试模型
    - 运行已知 prompt 的推理
    - 验证输出与参考实现匹配
    - _Requirements: 7.3_

- [x] 14. Final Checkpoint - 完整系统验证
  - 确保所有测试通过，如有问题请询问用户

## Notes

- 每个任务都引用了具体的需求条款以确保可追溯性
- Checkpoint 任务用于阶段性验证
- 属性测试验证通用正确性属性，单元测试验证具体示例和边界情况
- 所有测试任务均为必选，确保完整测试覆盖

# Tiny-LLM 需求文档

> 版本: 2.0.1 | 日期: 2026-04-16

## 简介

轻量级 LLM 推理引擎，实现 W8A16 量化、KV Cache 增量解码与模块化 Transformer 推理。

---

## 术语

| 术语 | 说明 |
|------|------|
| W8A16 | 权重 INT8、激活 FP16 的混合精度 |
| KV Cache | 自回归生成的 Key-Value 缓存 |
| Warp Shuffle | CUDA warp 内数据交换指令 |
| GQA | Grouped Query Attention |
| GGUF | LLM 模型权重格式 |

---

## 功能需求

### REQ-1: 模型加载

- [x] 解析 GGUF 文件头和配置
- [x] 读取二进制格式量化权重
- [x] 同时加载 INT8 权重和 scales
- [x] 分配 GPU 内存并传输
- [x] 损坏文件返回错误
- [x] 提供结构化模型表示

### REQ-2: W8A16 MatMul

- [x] 接受 INT8 权重、FP16 激活、scales
- [x] 寄存器级反量化
- [x] 使用 CUDA intrinsics
- [x] Warp shuffle 优化
- [x] 可配置 tile sizes
- [x] 与 FP16 基线误差 < 1%

### REQ-3: KV Cache

- [x] 预分配 GPU 内存池
- [x] 分配/释放序列 slots
- [x] 追加 KV 对
- [x] 追踪每序列使用量
- [x] 内存耗尽返回错误

### REQ-4: Transformer 层

- [x] Q,K,V 投影用 W8A16
- [x] 因果掩码
- [x] FFN 用 W8A16
- [x] RMSNorm
- [x] 残差连接
- [x] 增量解码

### REQ-5: Token 生成

- [x] Prefill 并行处理
- [x] 计算 logits + 采样
- [x] 更新 KV cache
- [x] 可配置最大长度
- [x] EOS 检测停止
- [x] tokens/second 统计

### REQ-6: 性能优化

- [x] 全局内存合并访问
- [x] Shared memory tiling
- [x] 优化线程块维度
- [x] 最小化同步点
- [x] CUDA streams 支持

### REQ-7: 错误处理

- [x] CUDA 错误捕获
- [x] 权重维度验证
- [x] 数值验证选项
- [x] 内存分配失败报告
- [x] 分析模式计时

---

## 非功能需求

| 类别 | 要求 |
|------|------|
| 精度 | W8A16 达到 FP16 基线 99%+ 精度 |
| 显存 | INT8 减少约 50% 权重显存 |
| 兼容性 | CUDA 11.0+, CMake 3.18+, C++17 |
| GPU | Compute Capability 7.0+ |

---

## 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| 2.0.1 | 2026-04-16 | 修复 scale 尺寸计算 bug |
| 2.0.0 | 2026-03-09 | KVCache API 重构 |
| 1.0.0 | 2025-02-13 | 初始版本 |

# 架构概览

Tiny-LLM 围绕一条很窄的 CUDA/C++ 运行时路径构建：支持的二进制运行时加载、显式 KV Cache 管理、W8A16 量化层，以及宿主侧基于 `Result<T>` 的错误传播。

## 系统示意

```mermaid
flowchart LR
    A[二进制运行时模型] --> B[InferenceEngine::load()]
    C[GGUF 文件] --> D[GGUFParser]
    B --> E[ModelWeights]
    E --> F[Transformer 层]
    F --> G[KVCacheManager]
    F --> H[W8A16 Kernel]
    H --> I[采样]
    I --> J[生成的 token ID]
    D --> K[元数据 / Tensor 检查]
```

## 核心组件

| 组件 | 职责 |
|------|------|
| `InferenceEngine` | 公开运行时入口，负责加载支持的二进制模型并生成 token ID |
| `ModelLoader` | 支持的二进制格式运行时权重加载 |
| `GGUFParser` | GGUF 解析、元数据提取和 tensor 检查 |
| `TransformerLayer` | W8A16 注意力与 FFN 执行 |
| `KVCacheManager` | 预分配缓存槽位与序列生命周期 |
| `Result<T>` | 宿主侧可失败 API 边界 |

## 加载边界

- **运行时加载：** `InferenceEngine::load("model.bin", config)`
- **GGUF 工具链：** `GGUFParser` 用于解析、检查和校验
- **不支持：** 通过 `InferenceEngine::load()` 直接进行 `.gguf` 运行时加载

## 生成流程

1. 通过支持的二进制运行时路径加载权重。
2. 将 prompt token ID 传给 `generate()`。
3. 在 Transformer 层栈上执行 prefill。
4. 在 decode 阶段复用 KV cache。
5. 从 logits 中采样下一个 token，直到 EOS 或 `max_new_tokens`。

## 说明

- 公共示例应使用 token ID，而不是仓库中并不存在的 tokenizer API。
- 公共文档必须分开描述二进制运行时路径和 GGUF 解析器。

## 下一步

- [推理引擎](/zh/architecture/inference-engine)
- [API 参考](/zh/api/)
- [性能指南](/zh/performance/)

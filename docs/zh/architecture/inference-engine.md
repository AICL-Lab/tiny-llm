# 推理引擎

`InferenceEngine` 是 Tiny-LLM 的公开运行时入口。它负责持有已加载的运行时权重、创建 KV cache、构建 Transformer 层栈，并驱动 prefill/decode 生成流程。

## 公共 API

```cpp
#include <tiny_llm/inference_engine.h>

namespace tiny_llm {

class InferenceEngine {
public:
    static Result<std::unique_ptr<InferenceEngine>> load(
        const std::string& model_path,
        const ModelConfig& config);

    Result<std::vector<int>> generate(
        const std::vector<int>& prompt_tokens,
        const GenerationConfig& config);

    const GenerationStats& getStats() const;
    void resetStats();
};

} // namespace tiny_llm
```

## 加载边界

- `InferenceEngine::load()` 使用支持的**二进制运行时格式**。
- `.gguf` 路径会被运行时路径以结构化错误拒绝。
- GGUF 解析与元数据提取属于 `GGUFParser`，不属于运行时快路径。

## 职责

1. 在运行时初始化前校验 `ModelConfig`。
2. 通过 `ModelLoader::loadBin()` 加载运行时权重。
3. 分配 CUDA stream 和可复用的 hidden/logit buffer。
4. 创建 `KVCacheManager` 与 `TransformerLayer` 层栈。
5. 先执行一次 prefill，再逐 token decode，直到 EOS 或生成上限。

## 输入与输出

- **输入：** prompt token ID（`std::vector<int>`）
- **输出：** 通过 `Result<std::vector<int>>` 返回生成的 token ID
- **统计：** prefill 时间、decode 时间、吞吐与 token 计数

## 失败模型

运行时表面会通过 `Result<T>` 返回配置无效、输入不受支持、分配失败和生成校验失败等错误。调用方在读取 `value()` 之前应先检查 `isErr()`。

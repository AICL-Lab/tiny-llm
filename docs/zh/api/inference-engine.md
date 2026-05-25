---
layout: doc
title: "InferenceEngine API"
description: "Tiny-LLM 的公开运行时 API"
---

# InferenceEngine API

## 类概览

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

    static int sampleGreedy(const half* logits, int vocab_size);
    static int sampleTemperature(const half* logits, int vocab_size, float temperature,
                                 unsigned seed = 0);
    static int sampleTopK(const half* logits, int vocab_size, int k, float temperature,
                          unsigned seed = 0);
    static int sampleTopP(const half* logits, int vocab_size, float p, float temperature,
                          unsigned seed = 0);
};

} // namespace tiny_llm
```

## 运行时加载

- `load("model.bin", config)` 面向支持的二进制运行时格式。
- 传入 `.gguf` 路径会返回结构化错误，说明当前不支持直接 GGUF 运行时加载。
- GGUF 的解析与检查能力由 `GGUFParser` 提供。

## 使用示例

```cpp
#include <iostream>
#include <tiny_llm/inference_engine.h>

int main() {
    using namespace tiny_llm;

    ModelConfig config;
    auto engine_result = InferenceEngine::load("model.bin", config);
    if (engine_result.isErr()) {
        std::cerr << engine_result.error() << '\n';
        return 1;
    }

    auto engine = std::move(engine_result.value());

    GenerationConfig generation;
    generation.max_new_tokens = 64;
    generation.temperature = 0.7f;
    generation.top_p = 0.9f;
    generation.do_sample = true;

    auto output = engine->generate({1, 15043, 29892}, generation);
    if (output.isErr()) {
        std::cerr << output.error() << '\n';
        return 1;
    }

    const auto& stats = engine->getStats();
    std::cout << "生成了 " << output.value().size()
              << " 个 token，速度 " << stats.tokens_per_second << " tok/s\n";
}
```

## 输入输出约定

| 方法 | 输入 | 输出 |
|------|------|------|
| `load()` | 运行时模型路径 + `ModelConfig` | `Result<std::unique_ptr<InferenceEngine>>` |
| `generate()` | prompt token ID + `GenerationConfig` | `Result<std::vector<int>>` |
| `getStats()` | 无 | `GenerationStats` |

## 说明

- Prompt 输入是 token ID，不是原始文本字符串。
- 在调用 `value()` 或 `error()` 前，应先检查 `isErr()`。

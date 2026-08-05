# 快速开始

本页展示与当前代码库一致的最小运行时路径。

## 1. 加载支持的运行时模型

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
}
```

## 2. 生成 token ID

```cpp
using namespace tiny_llm;

auto engine = std::move(engine_result.value());

GenerationConfig generation;
generation.max_new_tokens = 32;
generation.do_sample = true;
generation.temperature = 0.7f;

auto output = engine->generate({1, 15043, 29892}, generation);
if (output.isErr()) {
    std::cerr << output.error() << '\n';
    return 1;
}
```

## 3. 明确边界

- `InferenceEngine::load()` 使用支持的二进制运行时格式。
- `GGUFParser` 是 GGUF 解析与检查入口。
- 公共运行时 API 面向 token ID，而不是原始字符串。

## 下一步

- [架构概览](/architecture/)
- [API 参考](/api/)
- [安装说明](/guide/installation)

# API 参考

Tiny-LLM 在 `tiny_llm` 命名空间下暴露了一套很紧凑的公共 C++ 接口。

## 主要头文件

| 头文件 | 用途 |
|--------|------|
| `<tiny_llm/inference_engine.h>` | 运行时加载与 token 生成 |
| `<tiny_llm/model_loader.h>` | 宿主侧运行时模型加载辅助 |
| `<tiny_llm/gguf_parser.h>` | GGUF 解析、元数据提取、tensor 检查 |
| `<tiny_llm/kv_cache.h>` | KV cache 分配与序列管理 |
| `<tiny_llm/result.h>` | `Result<T>` 错误传播 |
| `<tiny_llm/types.h>` | 共享配置、权重与统计类型 |

## 快速示例

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
    auto output = engine->generate({1, 15043, 29892}, GenerationConfig{});
    if (output.isErr()) {
        std::cerr << output.error() << '\n';
        return 1;
    }
}
```

## 加载边界

- 用 `InferenceEngine::load()` 加载支持的二进制运行时格式。
- 需要 GGUF 解析或元数据访问时使用 `GGUFParser`。
- 不要假设公共接口里存在 tokenizer 或“输入字符串直接生成字符串”的 API。

## 参考页面

- [InferenceEngine](/api/inference-engine)
- [Result&lt;T&gt;](/api/result)
- [KVCacheManager](/api/kv-cache)
- [ModelConfig](/api/model-config)

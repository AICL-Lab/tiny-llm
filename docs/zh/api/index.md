# API 参考

欢迎使用 Tiny-LLM API 参考文档。

## 核心组件

| 组件 | 描述 |
|------|------|
| [InferenceEngine](/zh/api/inference-engine) | 主推理引擎类 |
| [ModelConfig](/zh/api/model-config) | 模型超参数 |
| [Result\<T\>](/zh/api/result) | 错误处理包装器 |
| [KVCacheManager](/zh/api/kv-cache) | 键值缓存管理 |

## 快速示例

```cpp
#include <tiny-llm/inference_engine.h>
#include <tiny-llm/model_config.h>
#include <tiny-llm/result.h>

using namespace tinyllm;

int main() {
    // 创建引擎
    InferenceEngine engine;

    // 加载模型
    Result<void> load_result = engine.load("model.bin");
    if (!load_result.ok()) {
        std::cerr << "错误: " << load_result.error() << std::endl;
        return 1;
    }

    // 生成文本
    GenerationConfig config;
    config.max_tokens = 100;
    config.temperature = 0.7f;

    Result<std::string> result = engine.generate("你好，世界！", config);
    if (result.ok()) {
        std::cout << result.value() << std::endl;
    }

    return 0;
}
```

## 错误处理

所有操作返回 `Result\<T\>` 进行安全的错误传播：

```cpp
Result\<T\> result = some_operation();
if (!result.ok()) {
    // 处理错误
    std::cerr << result.error() << std::endl;
} else {
    // 使用结果
    T value = result.value();
}
```

## 命名空间

所有公共 API 都在 `tinyllm` 命名空间中：

```cpp
using namespace tinyllm;
// 或
tinyllm::InferenceEngine engine;
```

## 头文件

| 头文件 | 描述 |
|--------|------|
| `<tiny-llm/inference_engine.h>` | 主引擎类 |
| `<tiny-llm/model_config.h>` | 配置结构体 |
| `<tiny-llm/result.h>` | Result 类型 |
| `<tiny-llm/kv_cache.h>` | 缓存管理 |
| `<tiny-llm/tokenizer.h>` | 分词工具 |

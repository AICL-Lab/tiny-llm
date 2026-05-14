# 快速开始

本教程将引导您完成使用 Tiny-LLM 进行首次推理。

## 前提条件

请确保您已经[安装 Tiny-LLM](/zh/guide/installation)。

## 基本用法

### 1. 加载模型

```cpp
#include <tiny-llm/inference_engine.h>

using namespace tinyllm;

int main() {
    // 创建推理引擎
    InferenceEngine engine;

    // 从文件加载模型
    Result<void> result = engine.load("model.bin");
    if (!result.ok()) {
        std::cerr << "加载模型失败: " << result.error() << std::endl;
        return 1;
    }

    return 0;
}
```

### 2. 运行推理

```cpp
#include <tiny-llm/inference_engine.h>
#include <iostream>

int main() {
    InferenceEngine engine;
    engine.load("model.bin");

    // 从提示词生成文本
    std::string prompt = "人工智能的未来是";
    Result<std::string> result = engine.generate(prompt, /*max_tokens=*/50);

    if (result.ok()) {
        std::cout << "生成结果: " << result.value() << std::endl;
    } else {
        std::cerr << "错误: " << result.error() << std::endl;
    }

    return 0;
}
```

### 3. 配置选项

```cpp
GenerationConfig config;
config.max_tokens = 100;
config.temperature = 0.7f;
config.top_p = 0.9f;
config.top_k = 50;

Result<std::string> result = engine.generate(prompt, config);
```

## 编译您的应用程序

```bash
# 编译您的应用程序
g++ -std=c++17 my_app.cpp -o my_app -I/path/to/tiny-llm/include -L/path/to/tiny-llm/build -ltinyllm
```

## 下一步

- [架构概览](/zh/architecture/inference-engine) - 了解内部原理
- [配置指南](/zh/guide/configuration) - 详细的配置选项
- [API 参考](/zh/api/) - 完整的 API 文档

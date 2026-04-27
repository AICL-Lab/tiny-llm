---
layout: default
title: "快速开始 — Tiny-LLM"
description: "几分钟内快速上手 Tiny-LLM 推理引擎"
nav_order: 2
---

# 快速开始

几分钟内完成 Tiny-LLM 推理引擎的安装与使用。

---

## 目录

- [前置要求](#前置要求)
- [安装步骤](#安装步骤)
- [快速示例](#快速示例)
- [模型格式](#模型格式)
- [下一步](#下一步)

---

## 前置要求

### 系统要求

| 组件 | 最低配置 | 推荐配置 |
|-----------|---------|-------------|
| CUDA Toolkit | 11.0 | 12.0+ |
| CMake | 3.18 | 3.25+ |
| C++ 编译器 | GCC 9+ / Clang 10+ | GCC 11+ / Clang 14+ |
| GPU 计算能力 | SM 7.0 (Volta) | SM 8.0+ (Ampere+) |
| GPU 显存 | 4 GB | 8 GB+ |

### 验证 GPU 支持

```bash
# 检查 CUDA 版本
nvcc --version

# 检查 GPU 计算能力
nvidia-smi --query-gpu=compute_cap --format=csv
# 输出应 >= 7.0
```

---

## 安装步骤

### 1. 克隆仓库

```bash
git clone https://github.com/LessUp/tiny-llm.git
cd tiny-llm
```

### 2. 配置构建

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
```

#### CMake 选项

| 选项 | 默认值 | 说明 |
|--------|---------|-------------|
| `CMAKE_BUILD_TYPE` | `Release` | 构建类型: Debug/Release/RelWithDebInfo |
| `BUILD_TESTS` | `ON` | 构建测试套件 |
| `CUDA_ARCH` | `native` | 目标 CUDA 架构 (如 `75;80;86`) |

### 3. 编译

```bash
make -j$(nproc)
```

### 4. 运行测试

```bash
ctest --output-on-failure
```

### 5. 运行 Demo

```bash
./tiny_llm_demo
```

---

## 快速示例

### 完整推理示例

```cpp
#include <tiny_llm/inference_engine.h>
#include <iostream>

int main() {
    // 1. 配置模型参数
    ModelConfig config;
    config.vocab_size = 32000;
    config.hidden_dim = 4096;
    config.num_layers = 32;
    config.num_heads = 32;
    config.num_kv_heads = 32;      // GQA: 现代模型使用 8
    config.head_dim = 128;
    config.intermediate_dim = 11008;
    config.max_seq_len = 2048;
    config.rope_theta = 10000.0f;
    
    // 2. 加载模型
    auto result = InferenceEngine::load("path/to/model.bin", config);
    if (result.isErr()) {
        std::cerr << "加载模型失败: " << result.error() << std::endl;
        return 1;
    }
    auto engine = std::move(result.value());
    
    // 3. 配置生成参数
    GenerationConfig gen_config;
    gen_config.max_new_tokens = 256;
    gen_config.temperature = 0.7f;
    gen_config.top_p = 0.9f;
    gen_config.top_k = 50;
    gen_config.do_sample = true;
    
    // 4. 生成文本
    std::vector<int> prompt = {1, 15043, 29892};  // "Hello," token
    auto output = engine->generate(prompt, gen_config);
    
    // 5. 查看性能统计
    const auto& stats = engine->getStats();
    std::cout << "生成了 " << stats.tokens_generated << " 个 token\n"
              << "速度: " << stats.tokens_per_second << " tok/s\n"
              << "显存峰值: " << stats.peak_memory_bytes / 1024 / 1024 << " MB\n";
    
    return 0;
}
```

### 直接使用 KV 缓存

```cpp
#include <tiny_llm/kv_cache.h>

// 创建缓存管理器
KVCacheConfig cache_config;
cache_config.num_layers = 32;
cache_config.num_heads = 32;
cache_config.head_dim = 128;
cache_config.max_seq_len = 2048;
cache_config.max_batch_size = 1;

// 使用工厂方法进行正确的错误处理
auto cache_result = KVCacheManager::create(cache_config);
if (cache_result.isErr()) {
    std::cerr << "创建缓存失败: " << cache_result.error() << std::endl;
    return 1;
}
auto kv_cache = std::move(cache_result.value());

// 分配序列
auto seq_result = kv_cache->allocateSequence(1024);
if (seq_result.isErr()) {
    std::cerr << "分配失败: " << seq_result.error() << std::endl;
    return 1;
}
int seq_id = seq_result.value();

// 在 Transformer 层中使用
for (auto& layer : layers) {
    layer.forward(hidden_states, *kv_cache, seq_id, position, stream);
}

// 所有层完成后，推进序列长度
kv_cache->advanceSeqLen(seq_id, 1);

// 完成后释放
kv_cache->releaseSequence(seq_id);
```

---

## 模型格式

### 自定义二进制格式

Tiny-LLM 目前使用自定义二进制格式，文件布局如下：

```
┌─────────────────┬─────────────────────────────────────┐
│ 头部 (256B)     │ 魔数、版本号、配置信息              │
├─────────────────┼─────────────────────────────────────┤
│ Token 词嵌入    │ [vocab_size, hidden_dim] FP16       │
├─────────────────┼─────────────────────────────────────┤
│ 第 0 层权重     │ Attention + FFN 权重 (INT8)         │
│                 │ 缩放因子 (FP16)                     │
├─────────────────┼─────────────────────────────────────┤
│ ...             │                                     │
├─────────────────┼─────────────────────────────────────┤
│ 第 N-1 层       │                                     │
├─────────────────┼─────────────────────────────────────┤
│ 输出 Norm       │ [hidden_dim] FP16                   │
│ LM Head         │ [hidden_dim, vocab_size] FP16       │
└─────────────────┴─────────────────────────────────────┘
```

### 创建模型文件

参考 [开发者指南](DEVELOPER) 了解如何将模型转换为 Tiny-LLM 格式。

---

## 下一步

- **[架构设计](ARCHITECTURE)** — 了解系统设计和组件
- **[API 参考](API)** — 完整 API 文档
- **[性能基准](BENCHMARKS)** — 性能特征
- **[故障排除](TROUBLESHOOTING)** — 常见问题和解决方案

---

**Languages**: [English](../en/QUICKSTART) | [中文](QUICKSTART) | [API 参考 →](API)

[← 返回首页](../../) | [架构设计 →](ARCHITECTURE)

---
layout: default
title: "快速开始 — Tiny-LLM"
description: "Tiny-LLM 推理引擎快速上手指南"
nav_order: 1
---

# 快速开始

几分钟内上手 Tiny-LLM。

---

## 前置要求

| 要求 | 版本 | 说明 |
|------|------|------|
| CUDA Toolkit | 11.0+ | 编译必需 |
| CMake | 3.18+ | 构建系统 |
| C++ 编译器 | C++17 | GCC 9+, Clang 10+, MSVC 2019+ |
| NVIDIA GPU | SM 7.0+ | Volta 架构或更新 |

### 验证 GPU 支持

检查 GPU 计算能力：

```bash
nvidia-smi --query-gpu=compute_cap --format=csv
# 输出应 >= 7.0
```

---

## 安装

### 1. 克隆仓库

```bash
git clone https://github.com/LessUp/tiny-llm.git
cd tiny-llm
```

### 2. 编译

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### CMake 选项

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `CMAKE_BUILD_TYPE` | `Release` | 构建类型: Debug/Release/RelWithDebInfo |
| `BUILD_TESTS` | `ON` | 编译测试套件 |
| `CUDA_ARCH` | `native` | CUDA 架构 (如 `75;80;86`) |

### 3. 运行测试

```bash
ctest --output-on-failure
```

### 4. 运行 Demo

```bash
./tiny_llm_demo
```

---

## 基础使用

### 简单推理

```cpp
#include <tiny_llm/inference_engine.h>
#include <iostream>

int main() {
    // 1. 配置模型
    ModelConfig config;
    config.vocab_size = 32000;
    config.hidden_dim = 4096;
    config.num_layers = 32;
    config.num_heads = 32;
    config.max_seq_len = 2048;
    
    // 2. 加载模型
    auto result = InferenceEngine::load("path/to/model.bin", config);
    if (result.isErr()) {
        std::cerr << "加载模型失败: " << result.error() << std::endl;
        return 1;
    }
    
    auto engine = std::move(result.value());
    
    // 3. 配置生成参数
    GenerationConfig gen_config;
    gen_config.max_new_tokens = 100;
    gen_config.temperature = 0.7f;
    gen_config.top_p = 0.9f;
    gen_config.do_sample = true;
    
    // 4. 生成文本
    std::vector<int> prompt = {1, 2, 3};  // Token IDs
    auto output = engine->generate(prompt, gen_config);
    
    // 5. 查看性能统计
    const auto& stats = engine->getStats();
    std::cout << "生成了 " << stats.tokens_generated 
              << " 个 token，耗时 " << stats.decode_time_ms << " ms"
              << " (速度: " << stats.tokens_per_second << " tok/s)" << std::endl;
    
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

KVCacheManager kv_cache(cache_config);

// 分配序列
auto seq_result = kv_cache.allocateSequence(1024);
if (seq_result.isErr()) {
    // 错误处理
}
int seq_id = seq_result.value();

// 在 Transformer 层中使用
layer.forward(hidden_states, kv_cache, seq_id, position);

// 所有层完成后推进序列长度
kv_cache.advanceSeqLen(seq_id, 1);

// 完成后释放
kv_cache.releaseSequence(seq_id);
```

---

## 构建自定义模型格式

目前，Tiny-LLM 支持自定义二进制格式。文件布局如下：

```
模型文件格式:
┌─────────────────┐
│  头部 (256B)    │
│  - magic        │
│  - version      │
│  - config       │
├─────────────────┤
│  词嵌入         │
│  [V, H] FP16    │
├─────────────────┤
│  第 0 层        │
│  - 权重...      │
├─────────────────┤
│  ...            │
├─────────────────┤
│  第 N 层        │
├─────────────────┤
│  输出 Norm      │
│  LM Head        │
└─────────────────┘
```

注意：完整的 GGUF 加载支持计划在未来的版本中实现。

---

## 故障排除

### 编译问题

| 问题 | 解决方案 |
|------|----------|
| `CUDA not found` | 设置 `CUDA_TOOLKIT_ROOT_DIR` 或确保 `nvcc` 在 PATH 中 |
| `CMake version too old` | 升级 CMake 或使用 pip: `pip install cmake` |
| `C++17 not supported` | 升级编译器 |

### 运行时问题

| 问题 | 解决方案 |
|------|----------|
| `CUDA out of memory` | 减小 `max_seq_len` 或 `num_layers` |
| `Illegal memory access` | 检查模型文件格式和维度 |
| `运行速度慢` | 确保 Release 构建: `-DCMAKE_BUILD_TYPE=Release` |

### Debug 构建

调试时使用：

```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON
make -j$(nproc)
CUDA_LAUNCH_BLOCKING=1 ./tiny_llm_demo
```

---

## 下一步

- 阅读 [API 参考](API) 获取详细 API 文档
- 查看 [架构设计](ARCHITECTURE) 了解系统设计
- 查看 [贡献指南](../../CONTRIBUTING) 参与项目
- 浏览 [更新日志](../../changelog/) 了解版本历史

---

**Languages**: [English](../en/QUICKSTART) | [中文](QUICKSTART)

[← 返回首页](../../) | [架构设计](ARCHITECTURE) | [API 参考](API)

# Tiny-LLM 推理引擎

[![CI](https://github.com/LessUp/tiny-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/tiny-llm/actions/workflows/ci.yml)
[![Pages](https://github.com/LessUp/tiny-llm/actions/workflows/pages.yml/badge.svg)](https://lessup.github.io/tiny-llm/)
[![Release](https://img.shields.io/github/v/release/LessUp/tiny-llm?include_prereleases)](https://github.com/LessUp/tiny-llm/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white)
![CMake](https://img.shields.io/badge/CMake-3.18+-064F8C?logo=cmake&logoColor=white)

[English](README.md) | 简体中文 | [文档](https://lessup.github.io/tiny-llm/) | [API 参考](https://lessup.github.io/tiny-llm/docs/zh/API)

轻量级、高性能 CUDA C++ 推理引擎，支持 Transformer 模型，具有 W8A16 量化、高效 KV 缓存管理和优化的 CUDA Kernel。

---

## ✨ 核心特性

| 特性 | 说明 | 状态 |
|---------|-------------|--------|
| **W8A16 量化** | INT8 权重 + FP16 激活，显存减少约 50% | ✅ 稳定 |
| **KV 缓存管理** | 高效的增量解码与序列管理 | ✅ 稳定 |
| **优化 CUDA Kernel** | Tensor Core INT8、共享内存 tiling、warp shuffle | ✅ 稳定 |
| **多采样策略** | 贪婪、温度、Top-k、Top-p 解码 | ✅ 稳定 |
| **完整测试** | GoogleTest + RapidCheck 基于属性的测试 | ✅ 稳定 |
| **双语文档** | 完整的中英文文档 | ✅ 完成 |

### 路线图

| 特性 | 状态 | 目标版本 |
|---------|--------|--------|
| GGUF 运行时加载 | 🚧 计划中 | v2.1 |
| PagedAttention | 📋 计划中 | v2.2 |
| 投机解码 | 🔬 研究中 | v2.3 |
| FP8 支持 | 🔬 研究中 | v3.0 |
| 多 GPU | 📋 计划中 | v3.0 |

---

## 🚀 快速开始

### 系统要求

| 组件 | 最低配置 | 推荐配置 |
|-----------|---------|-------------|
| NVIDIA GPU | SM 7.0 (Volta) | SM 8.0+ (Ampere+) |
| CUDA Toolkit | 11.0 | 12.0+ |
| CMake | 3.18 | 3.25+ |
| C++ 编译器 | GCC 9+ / Clang 10+ | GCC 11+ |

### 安装

```bash
git clone https://github.com/LessUp/tiny-llm.git
cd tiny-llm
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
ctest --output-on-failure
```

### 使用示例

```cpp
#include <tiny_llm/inference_engine.h>

ModelConfig config;
config.vocab_size = 32000;
config.hidden_dim = 4096;
config.num_layers = 32;

auto engine = InferenceEngine::load("model.bin", config).value();

GenerationConfig gen;
gen.max_new_tokens = 256;
gen.temperature = 0.7f;

auto output = engine.generate({1, 15043, 29892}, gen);
```

---

## 📚 文档

| 资源 | 说明 |
|----------|-------------|
| [📖 文档首页](https://lessup.github.io/tiny-llm/) | 项目文档首页 |
| [🚀 快速开始](https://lessup.github.io/tiny-llm/docs/zh/QUICKSTART) | 安装和使用指南 |
| [🏗️ 架构设计](https://lessup.github.io/tiny-llm/docs/zh/ARCHITECTURE) | 系统设计和组件 |
| [📘 API 参考](https://lessup.github.io/tiny-llm/docs/zh/API) | 完整 API 文档 |
| [⚡ 性能基准](https://lessup.github.io/tiny-llm/docs/zh/BENCHMARKS) | 性能测试数据 |
| [🔧 故障排除](https://lessup.github.io/tiny-llm/docs/zh/TROUBLESHOOTING) | 常见问题和解决方案 |
| [📝 更新日志](changelog/) | 版本历史 |

---

## 🔌 GPU 支持

| 架构 | 计算能力 | 状态 |
|--------------|-------------------|--------|
| Volta | SM 7.0, 7.5 | ✅ 支持 |
| Turing | SM 7.5 | ✅ 支持 |
| Ampere | SM 8.0, 8.6 | ✅ 优化 |
| Ada Lovelace | SM 8.9 | ✅ 优化 |
| Hopper | SM 9.0 | ✅ 支持 |

---

## 📁 项目结构

```
tiny-llm/
├── include/tiny_llm/    # 公共头文件
├── kernels/             # CUDA kernels (.cu, .cuh)
├── src/                 # 主机端实现 (.cpp)
├── tests/               # 单元测试和属性测试
├── docs/                # 文档 (中英)
│   ├── en/              # 英文文档
│   └── zh/              # 中文文档
├── changelog/           # 更新日志 (中英)
└── CMakeLists.txt       # 构建配置
```

---

## 🤝 贡献

我们欢迎贡献！请参阅 [CONTRIBUTING.md](CONTRIBUTING.md) 了解指南。

### 贡献者快速设置

```bash
# Fork 并克隆
git clone https://github.com/your-username/tiny-llm.git
cd tiny-llm

# 构建测试
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON
make -j$(nproc)

# 运行检查
ctest --output-on-failure
clang-format -i src/*.cpp tests/*.cu
```

---

## 📜 许可证

本项目采用 [MIT License](LICENSE)。

---

## 🙏 致谢

- 灵感来源于 [llama.cpp](https://github.com/ggerganov/llama.cpp) 和 [vLLM](https://github.com/vllm-project/vllm)
- 使用 [GoogleTest](https://github.com/google/googletest) 和 [RapidCheck](https://github.com/emil-e/rapidcheck) 构建

---

<p align="center">
  <a href="https://lessup.github.io/tiny-llm/">文档</a> •
  <a href="https://github.com/LessUp/tiny-llm/releases">发布</a> •
  <a href="https://github.com/LessUp/tiny-llm/issues">问题</a>
</p>

<p align="center">
  用 ❤️ 制作
</p>

# Tiny-LLM 推理引擎

[![CI](https://github.com/LessUp/tiny-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/tiny-llm/actions/workflows/ci.yml)
[![Pages](https://github.com/LessUp/tiny-llm/actions/workflows/pages.yml/badge.svg)](https://lessup.github.io/tiny-llm/)
[![Release](https://img.shields.io/github/v/release/LessUp/tiny-llm?include_prereleases)](https://github.com/LessUp/tiny-llm/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900?logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white)
![CMake](https://img.shields.io/badge/CMake-3.18+-064F8C?logo=cmake&logoColor=white)

[English](README.md) | 简体中文 | [文档](https://lessup.github.io/tiny-llm/) | [API 参考](https://lessup.github.io/tiny-llm/docs/)

轻量级 CUDA C++ 推理引擎，专注于 W8A16 量化、KV Cache 增量解码与模块化 Transformer 推理。

---

## 核心特性

| 特性 | 说明 | 状态 |
|------|------|------|
| **W8A16 量化** | INT8 权重 + FP16 激活，显存减少约 50% | ✅ 稳定 |
| **KV 缓存** | 高效的增量解码与序列管理 | ✅ 稳定 |
| **高性能 Kernel** | 共享内存 tiling、Warp shuffle 优化 | ✅ 稳定 |
| **多采样策略** | 贪婪、温度、Top-k、Top-p | ✅ 稳定 |
| **完整测试** | GoogleTest + RapidCheck 属性测试 | ✅ 稳定 |
| GGUF 运行时加载 | 运行时加载模型 | 🚧 计划中 |
| PagedAttention | 高效批处理 | 🚧 评估中 |

---

## 快速开始

```bash
git clone https://github.com/LessUp/tiny-llm.git
cd tiny-llm
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
ctest --output-on-failure
./tiny_llm_demo
```

**系统要求**: CUDA Toolkit 11.0+, CMake 3.18+, C++17 编译器, NVIDIA GPU (Compute Capability 7.0+)

---

## GPU 架构支持

| 架构 | 计算能力 | 状态 |
|------|---------|------|
| Volta | SM 7.0, 7.5 | ✅ 支持 |
| Turing | SM 7.5 | ✅ 支持 |
| Ampere | SM 8.0, 8.6 | ✅ 优化 |
| Ada Lovelace | SM 8.9 | ✅ 优化 |
| Hopper | SM 9.0 | ✅ 支持 |

---

## 架构概览

```
┌────────────────────────────────────────────────────────────┐
│                  InferenceEngine                            │
├────────────────────────────────────────────────────────────┤
│  模型加载器 ──► 权重 (INT8 + FP16 缩放因子)                 │
├────────────────────────────────────────────────────────────┤
│  Transformer 层 × N                                        │
│  ├── 注意力: W8A16 矩阵乘 + KV 缓存 + RoPE + 掩码          │
│  └── FFN: W8A16 矩阵乘 + SwiGLU                            │
├────────────────────────────────────────────────────────────┤
│  采样: 贪婪 / 温度 / Top-k / Top-p                         │
└────────────────────────────────────────────────────────────┘
```

---

## 项目结构

```
tiny-llm/
├── include/tiny_llm/    # 公共头文件
├── kernels/             # CUDA kernels (.cu, .cuh)
├── src/                 # 主机端实现 (.cpp)
├── tests/               # 单元测试和属性测试
├── docs/                # 文档 (中英文)
│   ├── en/              # 英文文档
│   └── zh/              # 中文文档
└── changelog/           # 更新日志 (中英文)
```

---

## 文档

| 资源 | 说明 |
|------|------|
| [📚 文档首页](https://lessup.github.io/tiny-llm/) | 项目文档首页 |
| [🚀 快速开始](https://lessup.github.io/tiny-llm/docs/zh/QUICKSTART) | 安装和基础使用 |
| [🏗️ 架构设计](https://lessup.github.io/tiny-llm/docs/zh/ARCHITECTURE) | 系统设计与组件 |
| [📖 API 参考](https://lessup.github.io/tiny-llm/docs/zh/API) | 完整 API 文档 |
| [📝 更新日志](https://lessup.github.io/tiny-llm/changelog/) | 版本历史 |
| [🤝 贡献指南](CONTRIBUTING.md) | 开发指南 |

---

## 性能亮点

| 优化 | 技术 | 收益 |
|------|------|------|
| 显存 | W8A16 量化 | 权重显存减少约 50% |
| 计算 | Tensor Core INT8 | 加速矩阵乘法 |
| 显存带宽 | Kernel 融合 | 减少数据搬运 |
| 延迟 | KV 缓存 | O(1) 增量解码 |

---

## 当前状态

**v2.0.1** — 核心运行时、缓存管理和测试框架已完成。Demo 验证 CUDA 可用性。

### 路线图

- [ ] 完整的 GGUF 运行时加载支持
- [ ] W8A16 量化的可配置 `group_size`
- [ ] Paged Attention 高效批处理
- [ ] 投机解码

---

## 贡献

我们欢迎贡献！请参阅我们的 [贡献指南](CONTRIBUTING.md)。

---

## 许可证

本项目采用 [MIT 许可证](LICENSE)。

---

## 致谢

- 灵感来源于 [llama.cpp](https://github.com/ggerganov/llama.cpp) 和 [vLLM](https://github.com/vllm-project/vllm)
- 使用 [GoogleTest](https://github.com/google/googletest) 和 [RapidCheck](https://github.com/emil-e/rapidcheck) 构建

---

<p align="center">
  用 ❤️ 制作
</p>

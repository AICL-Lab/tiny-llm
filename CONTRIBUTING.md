---
layout: default
title: "贡献指南 — Tiny-LLM"
description: "如何参与 Tiny-LLM 项目开发"
---

# 贡献指南

感谢你对 Tiny-LLM 项目的关注！欢迎通过 Issue 和 Pull Request 参与贡献。

---

## 开发流程

本项目遵循**规范驱动开发（Spec-Driven Development）**。在编写代码之前，必须先更新或创建相应的规范文档。

1. **更新/创建 Specs**：在 `/specs` 目录下更新产品需求、RFC 或 API 定义
2. Fork 本仓库
3. 创建特性分支：`git checkout -b feature/your-feature`
4. 进行更改并确保测试通过
5. 提交更改：`git commit -m "feat: add your feature"`
6. 推送分支：`git push origin feature/your-feature`
7. 创建 Pull Request

> 详细的 AI 助手工作流请参阅 [AGENTS.md](AGENTS.md)。

---

## 构建与测试

### 系统要求

- CUDA Toolkit 11.0+
- CMake 3.18+
- C++17 兼容编译器 (GCC 9+, Clang 10+, MSVC 2019+)
- NVIDIA GPU (Compute Capability 7.0+)

### 构建

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### 运行测试

```bash
ctest --output-on-failure
```

### 运行 Demo

```bash
./tiny_llm_demo
```

---

## 代码规范

### 格式化

- 使用 `.editorconfig` 中定义的缩进和格式规则
- 项目使用 clang-format 进行格式检查
- CI 会自动检查代码格式

### 代码风格

- C++17 标准特性
- RAII 资源管理
- `Result<T>` 错误处理（避免异常用于控制流）
- CUDA 错误使用 `CUDA_CHECK` 宏

### 测试要求

- 新增功能需附带单元测试
- 核心算法需附带属性测试 (RapidCheck)
- 确保所有现有测试通过

---

## 提交信息格式

推荐使用 [Conventional Commits](https://www.conventionalcommits.org/)：

| 前缀 | 说明 |
|------|------|
| `feat:` | 新功能 |
| `fix:` | Bug 修复 |
| `docs:` | 文档更新 |
| `perf:` | 性能优化 |
| `test:` | 测试相关 |
| `refactor:` | 代码重构 |
| `ci:` | CI/CD 变更 |

**示例**:
```
feat: add top-p sampling support
fix: correct scale tensor dimension calculation
docs: update API reference for KVCacheManager
```

---

## 项目结构

```
tiny-llm/
├── specs/               # 规范文档 (product, RFC, API, testing)
├── include/tiny_llm/    # 公共头文件
├── kernels/             # CUDA Kernel (.cu, .cuh)
├── src/                 # 主机端实现 (.cpp)
├── tests/               # 测试文件
├── docs/                # 文档
├── changelog/           # 更新日志
└── AGENTS.md            # AI 工作流定义
```

---

## 添加新功能

### 添加新 Kernel

1. 在 `kernels/` 创建 `.cu` 和 `.cuh` 文件
2. 在 `include/tiny_llm/` 添加公共接口声明
3. 在 `tests/` 添加单元测试和属性测试
4. 更新 `CMakeLists.txt`（如需要）

### 添加新测试

1. 单元测试使用 GoogleTest 宏
2. 属性测试使用 RapidCheck
3. 测试文件命名：`test_<module>.cpp` 或 `test_<module>.cu`

---

## 文档更新

- API 变更需更新 `/specs/api/` 和 `docs/` 目录
- 重要变更需添加 changelog 条目
- README 变更需同步英文和中文版本
- 新功能/修改功能需先更新 `/specs/product/` 或 `/specs/rfc/`

---

[← 返回首页](./) | [API 参考](docs/API) | [更新日志](changelog/)

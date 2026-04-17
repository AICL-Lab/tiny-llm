---
layout: default
title: "开发者指南 — Tiny-LLM"
description: "Tiny-LLM 开发环境配置和贡献指南"
nav_order: 5
---

# 开发者指南

开发环境配置、代码规范以及贡献指南。

---

## 目录

- [开发环境](#开发环境)
- [构建系统](#构建系统)
- [测试](#测试)
- [代码规范](#代码规范)
- [贡献指南](#贡献指南)

---

## 开发环境

### 前置要求

| 工具 | 最低版本 | 推荐版本 |
|------|---------|-------------|
| CUDA Toolkit | 11.0 | 12.0+ |
| CMake | 3.18 | 3.25+ |
| GCC | 9.4 | 11+ |
| Clang | 10 | 14+ |
| Python | 3.8 | 3.10+ |

### IDE 配置

#### VS Code

推荐扩展：
- `ms-vscode.cpptools` — C/C++ 扩展
- `llvm-vs-code-extensions.vscode-clangd` — Clangd 语言服务器
- `NVIDIA.nsight-vscode-edition` — CUDA 支持

```json
// .vscode/settings.json
{
    "cmake.configureSettings": {
        "CMAKE_EXPORT_COMPILE_COMMANDS": "ON"
    },
    "C_Cpp.default.configurationProvider": "ms-vscode.cmake-tools"
}
```

#### CLion

```
Settings → Build, Execution, Deployment → Toolchains
→ Add → CUDA (设置 CUDA 路径)
```

---

## 构建系统

### Debug 构建

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON
make -j$(nproc)
```

### Release 构建（带调试信息）

```bash
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
make -j$(nproc)
```

### Sanitizer 构建

```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug \
         -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined"
make -j$(nproc)
```

### 交叉编译

```bash
# 指定架构
cmake .. -DCUDA_ARCH="75;80;86"

# 所有常见架构
cmake .. -DCUDA_ARCH="70;75;80;86;89;90"
```

---

## 测试

### 运行所有测试

```bash
ctest --output-on-failure
```

### 运行特定测试

```bash
./tests/tiny_llm_test --gtest_filter="W8A16MatmulTest.*"
```

### 调试 CUDA Kernel

```bash
# 启用 CUDA launch blocking
CUDA_LAUNCH_BLOCKING=1 ./tests/tiny_llm_test

# CUDA memcheck
cuda-memcheck ./tests/tiny_llm_test

# Compute sanitizer (CUDA 11.1+)
compute-sanitizer --tool memcheck ./tests/tiny_llm_test
```

### 性能分析

```bash
# Nsight Compute
ncu -o profile.ncu-rep ./test_kernel

# Nsight Systems
nsys profile -o profile.qdrep ./test_app
```

---

## 代码规范

### C++ 代码规范

我们遵循 Google C++ 代码规范并做适当修改：

| 规则 | 约定 |
|------|------------|
| 命名 | 类用 `CamelCase`，函数/变量用 `snake_case` |
| 成员 | 私有成员尾部下划线: `member_` |
| 常量 | `kCamelCase` 用于常量，`SCREAMING_SNAKE` 用于宏 |
| 缩进 | 4 空格（不使用 Tab） |
| 行长度 | 100 字符 |

### 示例

```cpp
class MyClass {
public:
    explicit MyClass(int size);
    
    void doSomething(const std::string& input);
    
    int getSize() const { return size_; }
    
private:
    int size_;
    std::vector<float> data_;
};

namespace tiny_llm {

constexpr int kDefaultGroupSize = 128;

Result<float> computeValue(int input) {
    if (input < 0) {
        return Result<float>::err("Negative input");
    }
    return Result<float>::ok(std::sqrt(input));
}

}  // namespace tiny_llm
```

### 代码格式化

使用项目提供的 `.clang-format` 文件：

```bash
# 格式化单个文件
clang-format -i src/myfile.cpp

# 格式化所有源文件
find src tests kernels -name "*.cpp" -o -name "*.h" -o -name "*.cu" -o -name "*.cuh" \
    | xargs clang-format -i
```

---

## 贡献指南

### 工作流程

1. **Fork & Clone**
   ```bash
   git clone https://github.com/your-username/tiny-llm.git
   cd tiny-llm
   ```

2. **创建分支**
   ```bash
   git checkout -b feature/my-feature
   ```

3. **进行修改**
   - 编写代码
   - 添加测试
   - 更新文档

4. **提交**
   ```bash
   git commit -m "feat: add new feature"
   ```

5. **推送并创建 PR**
   ```bash
   git push origin feature/my-feature
   ```
   然后在 GitHub 上创建 Pull Request。

### Commit 消息格式

遵循 [Conventional Commits](https://www.conventionalcommits.org/) 规范：

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**类型**:
- `feat`: 新功能
- `fix`: Bug 修复
- `docs`: 文档变更
- `style`: 代码格式调整
- `refactor`: 代码重构
- `perf`: 性能优化
- `test`: 测试相关
- `ci`: CI/CD 变更
- `chore`: 维护任务

**示例**:
```
feat(attention): 添加多头注意力 kernel

实现优化的 attention_decode kernel，使用在线
softmax 改善数值稳定性。

fix(kvcache): 修复尺度维度计算错误

尺度张量使用错误的维度计算，导致 W8A16 矩阵乘中
的内存损坏。
```

### PR 清单

- [ ] 本地测试通过
- [ ] 代码符合规范（clang-format）
- [ ] 文档已更新
- [ ] Commit 消息符合规范
- [ ] PR 描述清楚说明变更

### 代码审查流程

1. 所有 PR 需要至少一次审查
2. CI 必须通过（格式检查、编译、测试）
3. 处理审查反馈
4. 根据需要 squash commits

---

**Languages**: [English](../en/DEVELOPER) | [中文](DEVELOPER)

[← API 参考](API) | [性能基准 →](BENCHMARKS)

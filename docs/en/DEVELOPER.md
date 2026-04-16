---
layout: default
title: "Developer Guide — Tiny-LLM"
description: "Development guide and contribution guidelines for Tiny-LLM"
nav_order: 5
---

# Developer Guide

Development environment setup and contribution guidelines.

---

## Table of Contents

- [Development Environment](#development-environment)
- [Build System](#build-system)
- [Testing](#testing)
- [Code Style](#code-style)
- [Contributing](#contributing)

---

## Development Environment

### Prerequisites

| Tool | Minimum | Recommended |
|------|---------|-------------|
| CUDA Toolkit | 11.0 | 12.0+ |
| CMake | 3.18 | 3.25+ |
| GCC | 9.4 | 11+ |
| Clang | 10 | 14+ |
| Python | 3.8 | 3.10+ |

### IDE Setup

#### VS Code

Recommended extensions:
- `ms-vscode.cpptools` — C/C++ extension
- `llvm-vs-code-extensions.vscode-clangd` — Clangd language server
- `NVIDIA.nsight-vscode-edition` — CUDA support

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
→ Add → CUDA (set CUDA path)
```

---

## Build System

### Debug Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON
make -j$(nproc)
```

### Release Build with Debug Info

```bash
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
make -j$(nproc)
```

### Sanitizer Build

```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug \
         -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined"
make -j$(nproc)
```

### Cross-Compilation

```bash
# For specific architecture
cmake .. -DCUDA_ARCH="75;80;86"

# For all common architectures
cmake .. -DCUDA_ARCH="70;75;80;86;89;90"
```

---

## Testing

### Run All Tests

```bash
ctest --output-on-failure
```

### Run Specific Test

```bash
./tests/tiny_llm_test --gtest_filter="W8A16MatmulTest.*"
```

### Debug CUDA Kernels

```bash
# Enable CUDA launch blocking for debugging
CUDA_LAUNCH_BLOCKING=1 ./tests/tiny_llm_test

# CUDA memcheck
cuda-memcheck ./tests/tiny_llm_test

# Compute sanitizer (CUDA 11.1+)
compute-sanitizer --tool memcheck ./tests/tiny_llm_test
```

### Profiling

```bash
# Nsight Compute
ncu -o profile.ncu-rep ./test_kernel

# Nsight Systems
nsys profile -o profile.qdrep ./test_app
```

---

## Code Style

### C++ Style Guide

We follow the Google C++ Style Guide with modifications:

| Rule | Convention |
|------|------------|
| Naming | `CamelCase` for classes, `snake_case` for functions/variables |
| Members | trailing underscore for private members: `member_` |
| Constants | `kCamelCase` for constants, `SCREAMING_SNAKE` for macros |
| Indent | 4 spaces (no tabs) |
| Line length | 100 characters |

### Example

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

### Formatting

Use `clang-format` with the project's `.clang-format` file:

```bash
# Format a file
clang-format -i src/myfile.cpp

# Format all source files
find src tests kernels -name "*.cpp" -o -name "*.h" -o -name "*.cu" -o -name "*.cuh" \
    | xargs clang-format -i
```

---

## Contributing

### Workflow

1. **Fork & Clone**
   ```bash
   git clone https://github.com/your-username/tiny-llm.git
   cd tiny-llm
   ```

2. **Create Branch**
   ```bash
   git checkout -b feature/my-feature
   ```

3. **Make Changes**
   - Write code
   - Add tests
   - Update documentation

4. **Commit**
   ```bash
   git commit -m "feat: add new feature"
   ```

5. **Push & PR**
   ```bash
   git push origin feature/my-feature
   ```
   Then create a Pull Request on GitHub.

### Commit Message Format

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, semicolons)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or correcting tests
- `ci`: CI/CD changes
- `chore`: Maintenance tasks

**Examples**:
```
feat(attention): add multi-head attention kernel

Implement optimized attention_decode kernel with online
softmax for improved numerical stability.

fix(kvcache): correct scale dimension calculation

The scale tensor was using incorrect dimension calculation
causing memory corruption in W8A16 matmul.
```

### PR Checklist

- [ ] Tests pass locally
- [ ] Code follows style guide (clang-format)
- [ ] Documentation updated
- [ ] Commit messages follow convention
- [ ] PR description explains changes

### Review Process

1. All PRs require at least one review
2. CI must pass (format check, build, tests)
3. Address review feedback
4. Squash commits if requested

---

**Languages**: [English](DEVELOPER) | [中文](../zh/DEVELOPER)

[← API Reference](API) | [Benchmarks →](BENCHMARKS)

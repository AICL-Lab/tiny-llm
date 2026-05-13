# Installation

This guide covers how to install Tiny-LLM on your system.

## System Requirements

### Hardware Requirements

- **GPU**: NVIDIA GPU with Compute Capability 7.0+ (Volta or newer)
- **Memory**: Minimum 8GB GPU memory for inference
- **Storage**: 2GB for build artifacts

### Software Requirements

- **Operating System**: Linux (Ubuntu 20.04+ recommended) or Windows 10+
- **CUDA Toolkit**: 11.0 or higher
- **CMake**: 3.18 or higher
- **C++ Compiler**: C++17 compatible (GCC 9+, Clang 10+, MSVC 2019+)

## Installation Steps

### 1. Install CUDA Toolkit

Ensure CUDA Toolkit 11.0+ is installed and `nvcc` is in your PATH:

```bash
nvcc --version
```

### 2. Clone the Repository

```bash
git clone https://github.com/shane0/tiny-llm.git
cd tiny-llm
```

### 3. Configure the Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
```

### 4. Build

```bash
cmake --build build -j$(nproc)
```

### 5. Run Tests (Optional)

```bash
ctest --test-dir build --output-on-failure
```

## Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `CMAKE_BUILD_TYPE` | `Release` | Build type (Release/Debug) |
| `BUILD_TESTS` | `OFF` | Build test suite |
| `BUILD_EXAMPLES` | `OFF` | Build example programs |
| `CUDA_ARCH` | Auto | CUDA architecture (e.g., `80;86`) |

## Troubleshooting

See the [Troubleshooting Guide](/en/guide/troubleshooting) for common issues.

## Next Steps

- [Quick Start Tutorial](/en/guide/quickstart) - Build your first inference
- [Architecture Overview](/en/guide/architecture) - Understand the internals

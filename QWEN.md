# Tiny-LLM — Project Context

## Project Overview

**Tiny-LLM** is a lightweight, high-performance CUDA C++ inference engine for Transformer models. It is designed for efficient LLM deployment on NVIDIA GPUs (Compute Capability 7.0+ / Volta or newer) with W8A16 quantization (INT8 weights + FP16 activations), achieving ~50% memory reduction compared to FP16 inference.

### Key Technologies

| Technology | Version | Purpose |
|------------|---------|---------|
| CUDA | 11.0+ | GPU acceleration |
| C++ | 17 | Host-side code |
| CMake | 3.18+ | Build system |
| spdlog | 1.12.0 | Logging (auto-fetched) |
| GoogleTest | 1.14.0 | Unit testing (auto-fetched) |
| RapidCheck | master | Property-based testing (auto-fetched) |

### Architecture Constraints

- **No exceptions for control flow** — errors use `Result<T>` monad
- **W8A16 quantization** — INT8 weights, FP16 activations
- **Zero external runtime dependencies** — pure CUDA C++
- **Spec-Driven Development** — specs are the single source of truth

---

## Directory Structure

```
tiny-llm/
├── include/tiny_llm/      # Public headers (11 files)
│   ├── inference_engine.h # Main engine class
│   ├── kv_cache.h         # KVCacheManager
│   ├── model_loader.h     # Model loading (binary + GGUF parser)
│   ├── gguf_parser.h      # GGUF format parser (consolidated types)
│   ├── transformer.h      # TransformerLayer forward pass
│   ├── types.h            # ModelConfig, ModelWeights, GenerationConfig
│   ├── result.h           # Result<T> monad for error handling
│   ├── cuda_utils.h       # CUDA_CHECK macros, CudaException, DeviceBuffer
│   ├── cuda_streams.h     # StreamPool, CudaEvent, KernelConfig
│   ├── logger.h           # Logger singleton + TLLM_* macros
│   └── validator.h        # Input validation utilities
├── kernels/               # CUDA kernels (.cu / .cuh)
│   ├── w8a16_matmul.cu    # Quantized matrix multiplication
│   ├── attention.cu       # Attention decode/prefill kernels
│   ├── rmsnorm.cu         # RMS normalization kernels
│   ├── elementwise.cu     # Add, SiLU, gather embeddings kernels
│   └── warp_utils.cuh     # Warp-level reduction utilities
├── src/                   # Host-side implementations (.cpp)
│   ├── main.cpp           # Demo entry point (excluded from lib)
│   ├── inference_engine.cpp
│   ├── model_loader.cpp
│   ├── gguf_parser.cpp
│   ├── kv_cache.cpp
│   ├── transformer.cpp
│   ├── types.cpp
│   └── logger.cpp
├── tests/                 # Unit & property-based tests (10 files)
│   ├── test_main.cpp      # Test entry point
│   ├── test_*.cpp         # CPU tests
│   └── test_*.cu          # GPU tests
├── specs/                 # Spec-Driven Development documents
│   ├── product/           # Feature requirements
│   ├── rfc/               # Architecture decisions (RFCs)
│   ├── api/               # API definitions (YAML)
│   ├── db/                # Data/model schemas
│   └── testing/           # BDD test specs (.feature)
├── website/               # GitHub Pages site (Jekyll)
│   ├── docs/              # Documentation (EN/ZH)
│   ├── changelog/         # Version history
│   ├── assets/            # CSS, JS, images
│   └── _config.yml        # Jekyll configuration
├── .github/workflows/     # CI/CD pipelines
│   ├── ci.yml             # Format check + build + test
│   ├── pages.yml          # Jekyll build + deploy
│   └── release.yml        # Release artifacts + GitHub release
├── CMakeLists.txt         # Build configuration
├── .clang-format          # LLVM-based C++17 formatting
├── .editorconfig          # Editor settings
└── README.md              # Project readme
```

---

## Building and Running

### Prerequisites

- NVIDIA GPU: SM 7.0+ (Volta or newer)
- CUDA Toolkit: 11.0+
- CMake: 3.18+
- C++ Compiler: GCC 9+ or Clang 10+

### Build Commands

```bash
# Clone
git clone https://github.com/LessUp/tiny-llm.git
cd tiny-llm

# Configure and build (Release, with tests by default)
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Run tests
ctest --output-on-failure

# Debug build with tests
cmake .. -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON
make -j$(nproc)
ctest --output-on-failure

# Build without tests
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF
make -j$(nproc)
```

### Build Targets

| Target | Type | Description |
|--------|------|-------------|
| `tiny_llm` | Static library | Core inference engine |
| `tiny_llm_demo` | Executable | Demo entry point (`src/main.cpp`) |
| `tiny_llm_tests` | Executable | GoogleTest + RapidCheck tests |

### Run Demo

```bash
./build/tiny_llm_demo
```

---

## Code Style and Conventions

### Formatting

- **Tool**: `clang-format` (LLVM-based, C++17)
- **Config**: `.clang-format` at project root
- **Column limit**: 100 characters
- **Indentation**: 4 spaces (2 spaces for YAML/JSON)
- **Pointer alignment**: Right (`int* ptr`)
- **Brace style**: Attach (K&R)
- **Include sorting**: Case-insensitive, preserve blocks

Run formatting check:
```bash
clang-format -i src/*.cpp include/tiny_llm/*.h kernels/*.cu tests/*.cpp tests/*.cu
```

### Naming Conventions

| Element | Style | Example |
|---------|-------|---------|
| Classes | PascalCase | `InferenceEngine`, `KVCacheManager` |
| Functions | camelCase | `generate()`, `loadGGUF()` |
| Member variables | snake_case with trailing `_` | `config_`, `weights_` |
| Constants / macros | UPPER_SNAKE_CASE | `TLLM_INFO`, `BIN_MAGIC` |
| Namespaces | snake_case | `tiny_llm` |
| Files | snake_case | `inference_engine.h`, `w8a16_matmul.cu` |

### Error Handling

Use `Result<T>` monad — **never throw exceptions for control flow**:

```cpp
Result<ModelWeights> loadModel(const std::string& path) {
    if (!fileExists(path)) {
        return Result<ModelWeights>::err("File not found: " + path);
    }
    // ...
    return Result<ModelWeights>::ok(weights);
}

// Usage
auto result = loadModel("model.bin");
if (result.isErr()) {
    TLLM_ERROR("Failed to load model: {}", result.error());
    return;
}
auto weights = result.value();
```

### Logging

Use the `TLLM_*` macros from `logger.h`:

```cpp
TLLM_INFO("Model loaded successfully: {}", model_path);
TLLM_WARN("KV cache full, evicting oldest sequence");
TLLM_ERROR("CUDA kernel failed: {}", cudaGetErrorString(err));
TLLM_DEBUG("Token generated: {}", token_id);
```

---

## Testing Practices

### Test Framework

- **GoogleTest** — unit tests, fixture-based tests
- **RapidCheck** — property-based testing (generative tests)

### Test File Conventions

- Test files go in `tests/`
- Naming: `test_<module>.cpp` or `test_<module>.cu`
- `.cpp` for CPU tests, `.cu` for GPU tests
- Each test file should have a corresponding header/implementation

### Running Tests

```bash
# Run all tests
ctest --output-on-failure

# Run specific test by name
ctest -R test_kv_cache

# Run with verbose output
ctest --output-on-failure -V

# Run tests with timeout (seconds)
ctest --output-on-failure --timeout 300
```

---

## Spec-Driven Development (SDD)

This project follows **Spec-Driven Development**. Before writing code:

1. **Read specs** in `/specs/` — they are the single source of truth
2. **Update specs first** if creating new features or changing APIs
3. **Implement code** that adheres to spec definitions
4. **Write tests** based on spec acceptance criteria

### Spec Directories

| Directory | Purpose |
|-----------|---------|
| `specs/product/` | Feature requirements and acceptance criteria |
| `specs/rfc/` | Technical design decisions (RFC format) |
| `specs/api/` | API interface definitions (YAML) |
| `specs/db/` | Data model schemas |
| `specs/testing/` | BDD test case specifications (.feature) |

### Spec Naming

| Type | Pattern | Example |
|------|---------|---------|
| Product | `<feature-name>.md` | `tiny-llm-inference-engine.md` |
| RFC | `<NNNN>-<short-title>.md` | `0001-core-architecture.md` |
| API | `<feature-name>.yaml` | `inference-engine.yaml` |
| Testing | `<feature-name>.feature` | `inference-engine.feature` |

---

## CI/CD

### Workflows

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `ci.yml` | Push/PR to main | Format check → Build → Test |
| `pages.yml` | Push to `website/` | Build & deploy GitHub Pages |
| `release.yml` | Tag `v*.*.*` or manual | Create release with artifacts |

### Environment Variables

- `FORCE_JAVASCRIPT_ACTIONS_TO_NODE24: true` — set in all workflows to avoid Node.js 20 deprecation

---

## Key Components

### InferenceEngine

Main entry point. Handles model loading, prompt prefilling, token generation, and sampling.

```cpp
auto engine = InferenceEngine::load("model.bin", config).value();
auto tokens = engine.generate(prompt_tokens, gen_config).value();
```

### KVCacheManager

Manages key-value cache for incremental decoding. Supports dynamic sequence allocation, explicit length advancement, and cache eviction.

```cpp
KVCacheManager cache(config);
auto seq = cache.allocateSequence(64).value();
cache.updateLength(seq.id, new_length);
cache.releaseSequence(seq.id);
```

### W8A16 MatMul

Quantized matrix multiplication: INT8 weights × FP16 input → FP32 output. Uses shared memory tiling and warp-level reductions for performance.

### Result<T>

Monadic error handling type. Key methods: `isOk()`, `isErr()`, `value()`, `error()`, `valueOr(default)`, `map(f)`, `flatMap(f)`.

---

## Useful Commands

```bash
# Check formatting
clang-format --dry-run --Werror $(find . -name '*.cpp' -o -name '*.h' -o -name '*.cu' -o -name '*.cuh' | grep -v build)

# Generate compile_commands.json for IDE
cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# Clean build
rm -rf build/

# Run tests with Google Test filter
./build/tiny_llm_tests --gtest_filter="*KVCache*"

# Check CUDA availability
nvcc --version
```

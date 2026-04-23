---
name: verify
description: Run format check, build, and tests to verify changes are correct. Use before committing or after significant edits.
---

Run full verification for Tiny-LLM:

## Steps

1. **Format check** — Verify C++/CUDA formatting:
   ```bash
   find . -type f \( -name '*.cpp' -o -name '*.h' -o -name '*.cu' -o -name '*.cuh' \) \
     ! -path './build/*' | xargs clang-format-18 --dry-run --Werror
   ```

2. **Build** — Compile in release mode:
   ```bash
   mkdir -p build && cd build && \
   cmake .. -DCMAKE_BUILD_TYPE=Release -DCUDAHOSTCXX=/usr/bin/g++-10 && \
   make -j$(nproc)
   ```

3. **Test** — Run all tests:
   ```bash
   cd build && ctest --output-on-failure --timeout 300
   ```

## On Failure

- If format check fails: run `clang-format-18 -i <files>` to fix
- If build fails: check GCC version (must be GCC 10 for CUDA)
- If tests fail: run specific test with `./tiny_llm_tests --gtest_filter="TestSuite.TestName"`

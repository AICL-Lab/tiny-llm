# 故障排除

本指南介绍常见问题及其解决方案。

## 安装问题

### 找不到 CUDA

**错误**: `Could not find CUDA`

**解决方案**:
```bash
# 检查 CUDA 安装
nvcc --version

# 如需设置 CUDA 路径
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### CMake 版本过旧

**错误**: `CMake 3.18 or higher is required`

**解决方案**:
```bash
# 安装较新版本的 CMake
pip install cmake --upgrade
# 或
sudo apt install cmake
```

### 编译器不支持 C++17

**错误**: `C++17 feature not supported`

**解决方案**:
```bash
# 安装 GCC 9+
sudo apt install gcc-9 g++-9

# 设置为默认
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 90
```

## 构建问题

### NVCC 内存不足

**错误**: `nvcc fatal : Out of memory`

**解决方案**:
```bash
# 减少并行任务数
cmake --build build -j2  # 而不是 -j$(nproc)
```

### 未定义符号错误

**错误**: `undefined reference to 'cublas...'`

**解决方案**:
```bash
# 确保 CUDA 库已链接
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
```

## 运行时问题

### CUDA 内存不足

**错误**: `CUDA out of memory`

**解决方案**:
```cpp
// 减少上下文长度
config.max_seq_len = 1024;

// 减少批处理大小
config.max_batch_size = 1;
```

### 生成速度慢

**症状**: Token 生成非常缓慢

**解决方案**:

1. 检查 GPU 利用率：
   ```bash
   nvidia-smi dmon
   ```

2. 验证模型在 GPU 上：
   ```cpp
   std::cout << engine.device() << std::endl;  // 应该输出 "cuda:0"
   ```

3. 启用优化：
   ```cpp
   config.enable_flash_attention = true;
   config.enable_cuda_graphs = true;
   ```

### 输出不正确

**症状**: 生成的文本是乱码或错误

**解决方案**:

1. 验证模型文件完整性：
   ```bash
   sha256sum model.bin
   ```

2. 检查词汇表大小是否匹配：
   ```cpp
   std::cout << engine.config().vocab_size << std::endl;
   ```

3. 确保分词器正确：
   ```cpp
   // 验证分词
   auto tokens = engine.encode("你好");
   for (auto t : tokens) std::cout << t << " ";
   ```

## 量化问题

### 精度下降

**症状**: 量化后生成质量差

**解决方案**:

1. 使用更小的组大小：
   ```cpp
   quant_config.group_size = 64;  // 而不是 128
   ```

2. 验证量化是否成功：
   ```bash
   ./build/bin/tinyllm-quant --verify model.bin
   ```

### 量化失败

**错误**: `Quantization failed: invalid weights`

**解决方案**:
- 确保输入模型是有效的 FP16/FP32
- 检查 NaN/Inf 值：
  ```bash
  python -c "import torch; m = torch.load('model.bin'); print(torch.isnan(m).any())"
  ```

## 平台特定问题

### Linux

**GPU 权限被拒绝**:
```bash
sudo usermod -aG video $USER
# 登出并重新登录
```

### Windows

**找不到 DLL**:
- 将 CUDA bin 目录添加到 PATH
- 将所需的 DLL 复制到可执行文件目录

## 获取帮助

如果您无法解决问题：

1. 查看现有的 [GitHub Issues](https://github.com/shane0/tiny-llm/issues)
2. 搜索 [Discussions](https://github.com/shane0/tiny-llm/discussions)
3. 创建新的 issue，包含：
   - 错误消息和堆栈跟踪
   - 您的系统信息（`nvidia-smi`、`nvcc --version`）
   - 重现步骤

## 下一步

- [性能指南](/zh/guide/performance) - 优化技术
- [API 参考](/zh/api/) - 完整的 API 文档

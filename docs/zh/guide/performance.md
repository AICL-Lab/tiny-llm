# 性能

本指南介绍 Tiny-LLM 的性能优化技术。

## 基准测试

### 运行基准测试

```bash
./build/bin/tinyllm-bench --model model.bin --prompt "你好，世界！"
```

### 关键指标

| 指标 | 描述 |
|------|------|
| **Tokens/秒** | 生成吞吐量 |
| **首个 Token 时间** | 首个 token 的延迟 |
| **内存使用** | GPU 内存消耗 |
| **利用率** | GPU 计算利用率 |

## 优化技术

### 1. KV 缓存调优

```cpp
KVCacheConfig config;
config.max_batch_size = 1;     // 单序列
config.max_seq_len = 2048;      // 满足您的需求
config.enable_swapping = false; // 单 GPU 时禁用
```

### 2. 批处理大小

对于吞吐量关键的应用：

```cpp
// 批量处理多个序列
engine.setBatchSize(8);
```

### 3. Flash Attention

启用 Flash Attention 以加速推理：

```cpp
config.enable_flash_attention = true;
```

### 4. CUDA Graphs

减少内核启动开销：

```cpp
config.enable_cuda_graphs = true;
```

## 内存优化

### 内存分布

| 组件 | 内存 (LLaMA-7B) |
|------|-----------------|
| 模型权重 (INT8) | ~3.5 GB |
| KV 缓存 (2048 上下文) | ~1.0 GB |
| 激活值 | ~0.5 GB |
| **总计** | ~5.0 GB |

### 减少内存使用

1. **减少上下文长度**:
   ```cpp
   config.max_seq_len = 1024;  // KV 缓存减半
   ```

2. **启用 KV 缓存卸载**:
   ```cpp
   config.enable_swapping = true;
   ```

3. **使用较小的批处理大小**:
   ```cpp
   config.max_batch_size = 1;
   ```

## 性能指南

### GPU 选择

| GPU | 推荐用途 |
|-----|----------|
| RTX 3060 (12GB) | 小型模型 (7B) |
| RTX 4090 (24GB) | 中型模型 (13B-30B) |
| A100 (40GB) | 大型模型 (65B+) |
| H100 (80GB) | 最大型模型 |

### 软件配置

1. **使用 CUDA 12+** 以获得最佳性能
2. **启用 P-State 0** 以获得最高时钟频率：
   ```bash
   sudo nvidia-smi -i 0 -pl 300  # 设置功率限制
   ```
3. **禁用 ECC** 以获得稍多内存：
   ```bash
   sudo nvidia-smi -e 0
   ```

## 故障排除性能问题

### Token 生成速率低

1. 检查 GPU 利用率：`nvidia-smi dmon`
2. 验证 CUDA 版本兼容性
3. 确保模型已加载到 GPU
4. 检查是否存在 CPU 瓶颈

### 内存错误

1. 减少上下文长度
2. 减少批处理大小
3. 启用 KV 缓存交换

## 下一步

- [故障排除指南](/zh/guide/troubleshooting) - 常见问题
- [API 参考](/zh/api/) - 完整的 API 文档

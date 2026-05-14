# 架构设计

Tiny-LLM 推理引擎的系统架构和设计文档。

## 概述

Tiny-LLM 是一个高性能 CUDA C++ 推理引擎，专为高效的 Transformer 模型推理而设计。其核心特性包括：

| 特性 | 技术 | 收益 |
|---------|------------|---------|
| **W8A16 量化** | INT8 权重 + FP16 激活 | 显存减少约 50% |
| **高效 KV 缓存** | 增量解码与序列管理 | O(1) 自回归步进 |
| **优化 Kernel** | Tensor Core INT8、共享内存 tiling | 最大吞吐 |
| **模块化设计** | 清晰职责分离 | 易于扩展和测试 |

---

## 系统架构

```mermaid
flowchart TB
    subgraph Input["📥 输入"]
        A["模型文件<br/>.gguf/.bin"] --> B["GGUF 解析器"]
        P["Prompt Tokens"] --> C["分词"]
    end

    subgraph Engine["⚙️ InferenceEngine"]
        B --> D["模型加载器"]
        D --> E["InferenceEngine"]
        C --> E
        E --> F["Transformer 层栈"]
    end

    subgraph Transformer["🔄 Transformer 层"]
        F --> G["嵌入层"]
        G --> H["Layer 1..N"]
        H --> I["LM Head"]
    end

    subgraph KVCache["💾 KV 缓存"]
        H <--> J["KV Cache 管理器"]
        J --> K["预分配槽位"]
    end

    subgraph Output["📤 输出"]
        I --> L["采样"]
        L --> M["生成的 Tokens"]
    end

    style A fill:#00D4AA,stroke:#00C49A,color:#fff
    style E fill:#76B900,stroke:#5a8f00,color:#fff
    style J fill:#F59E0B,stroke:#d97706,color:#fff
    style L fill:#8B5CF6,stroke:#7c3aed,color:#fff
```

---

## 核心组件

### 1. InferenceEngine

模型推理的主入口。

```cpp
class InferenceEngine {
public:
    // 从磁盘加载模型
    static Result\<std::unique_ptr\<InferenceEngine\>\> load(
        const std::string& path, const ModelConfig& config);
    
    // 完整的生成流程
    std::vector<int> generate(
        const std::vector<int>& prompt, 
        const GenerationConfig& config);
    
    // 统计和性能分析
    const GenerationStats& getStats() const;
    void resetStats();
};
```

**核心职责**:
- 模型生命周期管理
- Prefill/decode 编排
- Token 采样和生成循环
- 性能分析

### 2. KV Cache 管理器

用于自回归生成的高效键值缓存。

```mermaid
flowchart LR
    subgraph CacheStructure["KV 缓存内存布局"]
        K["K 缓存<br/>[batch, layers, seq, heads, dim]"]
        V["V 缓存<br/>[batch, layers, seq, heads, dim]"]
    end

    subgraph Operations["操作"]
        A1["allocateSequence()"]
        A2["appendKV()"]
        A3["getCache()"]
        A4["advanceSeqLen()"]
    end

    A1 --> CacheStructure
    A2 --> CacheStructure
    A3 --> CacheStructure
    A4 --> CacheStructure

    style K fill:#76B900,stroke:#5a8f00,color:#fff
    style V fill:#76B900,stroke:#5a8f00,color:#fff
```

### 3. W8A16 量化

仅权重的 INT8 量化，使用 FP16 激活。

```mermaid
flowchart TB
    subgraph QuantScheme["量化方案"]
        W["权重: INT8<br/>[rows, cols]"]
        S["缩放: FP16<br/>[rows/128, cols]"]
        O["输出: FP16<br/>= dequant(W, S) @ Act"]
    end

    W --> O
    S --> O

    subgraph GroupWise["组级布局"]
        G0["group 0 (128 元素) → scales[0]"]
        G1["group 1 (128 元素) → scales[1]"]
        GN["..."]
    end

    style W fill:#8B5CF6,stroke:#7c3aed,color:#fff
    style S fill:#F59E0B,stroke:#d97706,color:#fff
    style O fill:#00D4AA,stroke:#00C49A,color:#fff
```

**优势**:
- 权重显存减少 50%
- 不量化激活 (保持精度)
- Ampere+ 上高效的 INT8 Tensor Core 利用

---

## 数据流

### Prefill 阶段 (Prompt 处理)

```mermaid
sequenceDiagram
    participant User as 用户
    participant Engine as InferenceEngine
    participant Trans as Transformer
    participant KV as KV 缓存
    participant Sample as 采样

    User->>Engine: generate(prompt, config)
    Engine->>Engine: tokenize(prompt)
    Engine->>Trans: forward(tokens, prefill=true)
    
    loop 每一层
        Trans->>Trans: RMSNorm
        Trans->>Trans: QKV 投影 (W8A16)
        Trans->>Trans: RoPE
        Trans->>Trans: Attention Prefill
        Trans->>KV: store_kv(layer, k, v)
        Trans->>Trans: Output 投影
        Trans->>Trans: 残差
    end
    
    Trans-->>Engine: logits
    Engine->>Sample: sample(logits)
    Sample-->>Engine: first_token
    Engine-->>User: first_token
```

### Decode 阶段 (Token 生成)

```mermaid
sequenceDiagram
    participant User as 用户
    participant Engine as InferenceEngine
    participant Trans as Transformer
    participant KV as KV 缓存
    participant Sample as 采样

    loop 每个新 token
        User->>Engine: continue generation
        Engine->>Trans: forward(next_token, prefill=false)
        
        loop 每一层
            Trans->>Trans: RMSNorm
            Trans->>Trans: QKV 投影 (W8A16)
            Trans->>Trans: RoPE
            Trans->>KV: get_cache(layer)
            Trans->>Trans: Attention Decode
            Trans->>KV: append_kv(layer, k, v)
            Trans->>Trans: Output 投影
            Trans->>Trans: 残差
        end
        
        Trans-->>Engine: logits
        Engine->>Sample: sample(logits)
        Sample-->>Engine: next_token
    end
    
    Engine-->>User: full_output
```

---

## 内存布局

### 权重存储

```mermaid
block-beta
    columns 1
    
    block:Header["模型权重"]
        A["Token 嵌入<br/>FP16 [vocab, hidden]"]
    end
    
    block:Layers["层权重 (×N)"]
        B1["注意力<br/>INT8 q/k/v/o_proj"]
        B2["FFN<br/>INT8 gate/up/down"]
        B3["缩放<br/>FP16 每组"]
    end
    
    block:Output["输出"]
        C["LM Head<br/>FP16 [hidden, vocab]"]
    end
```

### 激活缓存

| 缓存 | 形状 | 数据类型 | 大小 (B=1, S=2048, H=4096) |
|--------|-------|-------|---------------------------|
| 隐藏状态 | [B, S, H] | FP16 | 16 MB |
| 注意力输出 | [B, heads, S, head_dim] | FP16 | 16 MB |
| QKV | [B, S, 3×H] | FP16 | 48 MB |
| FFN 中间结果 | [B, S, intermediate_dim] | FP16 | 44 MB |

---

## 性能优化

### 内存优化

| 技术 | 实现 | 收益 |
|-----------|----------------|---------|
| W8A16 量化 | 每组 INT8 权重 + FP16 缩放 | 权重显存减少 50% |
| KV Cache 分页 | 预分配 + 序列管理 | 高效批处理 |
| 激活复用 | 原地操作 | 减少分配 |

### 计算优化

| 技术 | 应用 | 加速比 |
|-----------|-------------|---------|
| Tensor Cores | INT8 矩阵乘 (Ampere+) | 2-4× vs FP16 |
| Warp Shuffle | 归约 | 消除共享内存 |
| 向量化加载 | 128-bit 内存访问 | 更好带宽 |
| Kernel 融合 | RMSNorm+Resid, SiLU+Mul | 减少内核启动 |

### Kernel 性能

```mermaid
xychart-beta
    title "Kernel 吞吐量 (7B 模型, A100)"
    x-axis ["w8a16_matmul", "attn_decode", "attn_prefill", "rmsnorm", "rope"]
    y-axis "吞吐量" 0-->100
    bar [80, 65, 75, 95, 98]
```

---

## 设计原则

```mermaid
mindmap
  root((设计原则))
    模块化
      清晰接口
      职责分离
      插件架构
    类型安全
      Result&lt;T&gt; 错误处理
      强类型
      无异常
    RAII
      自动 GPU 显存管理
      流管理
      资源清理
    可测试性
      单元测试
      基于属性测试
      CUDA 测试
    可扩展性
      新 Kernel
      采样策略
      模型格式
```

---

## 下一步

- [量化详解](./quantization) - W8A16 实现深入解析
- [KV 缓存](./kv-cache) - 缓存管理策略
- [CUDA Kernel](./cuda-kernels) - Kernel 优化技术
- [API 参考](../api/inference-engine) - 完整 API 文档

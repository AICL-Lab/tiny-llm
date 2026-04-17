---
layout: page
title: Tiny-LLM
description: High-Performance CUDA Inference Engine for LLMs — W8A16 quantization, KV Cache, and optimized CUDA kernels
nav_order: 1
permalink: /
---

<!-- Hero Section -->
<div class="hero-section">
  <div class="hero-content">
    <div class="hero-badge">
      <span class="badge-highlight">v{{ site.version }} Released</span>
      <a href="{{ site.baseurl }}/changelog/" class="changelog-link">See what's new</a>
    </div>

    <h1 class="hero-title">
      High-Performance<br>
      <span class="gradient-text">CUDA Inference Engine</span>
    </h1>

    <p class="hero-description">
      Lightweight, efficient LLM inference with <strong>W8A16 quantization</strong>,
      <strong>KV Cache optimization</strong>, and hand-tuned <strong>CUDA kernels</strong>.
      Built for speed. Designed for developers.
    </p>

    <div class="hero-actions">
      <a href="{{ site.baseurl }}/docs/en/QUICKSTART" class="btn btn-primary">
        Get Started
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M5 12h14M12 5l7 7-7 7"/>
        </svg>
      </a>
      <a href="{{ site.repo_url }}" class="btn btn-secondary">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
          <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
        </svg>
        View on GitHub
      </a>
    </div>

    <div class="hero-stats">
      <div class="stat">
        <span class="stat-value">~50%</span>
        <span class="stat-label">Memory Reduction</span>
      </div>
      <div class="stat-separator">|</div>
      <div class="stat">
        <span class="stat-value">CUDA</span>
        <span class="stat-label">Native Implementation</span>
      </div>
      <div class="stat-separator">|</div>
      <div class="stat">
        <span class="stat-value">W8A16</span>
        <span class="stat-label">Quantization</span>
      </div>
    </div>
  </div>

  <div class="hero-visual">
    <div class="code-window">
      <div class="code-header">
        <div class="code-dots">
          <span></span>
          <span></span>
          <span></span>
        </div>
        <span class="code-filename">example.cpp</span>
      </div>
      <div class="code-content">
{% highlight cpp %}
#include <tiny_llm/inference_engine.h>

// Configure model
ModelConfig config;
config.vocab_size = 32000;
config.hidden_dim = 4096;
config.num_layers = 32;

// Load with W8A16 weights
auto engine = InferenceEngine::load("model.bin", config).value();

// Generate with KV cache
GenerationConfig gen;
gen.max_new_tokens = 256;
gen.temperature = 0.7f;

auto output = engine.generate(prompt, gen);
{% endhighlight %}
      </div>
    </div>
  </div>
</div>

---

## Features

<div class="features-grid">
  <div class="feature-card">
    <div class="feature-icon">&#9889;</div>
    <h3>W8A16 Quantization</h3>
    <p>INT8 weights with FP16 activations deliver <strong>~50% memory reduction</strong> while maintaining inference quality. Optimized for modern GPUs with Tensor Core support.</p>
    <span class="badge badge-stable">Stable</span>
  </div>

  <div class="feature-card">
    <div class="feature-icon">&#128190;</div>
    <h3>Efficient KV Cache</h3>
    <p>State-of-the-art key-value cache management with <strong>O(1) incremental decoding</strong>. Supports dynamic sequence allocation and explicit length advancement.</p>
    <span class="badge badge-stable">Stable</span>
  </div>

  <div class="feature-card">
    <div class="feature-icon">&#128295;</div>
    <h3>Optimized CUDA Kernels</h3>
    <p>Hand-tuned kernels featuring shared memory tiling, warp shuffle reductions, and memory coalescing. Up to <strong>80% Tensor Core utilization</strong> on Ampere GPUs.</p>
    <span class="badge badge-stable">Stable</span>
  </div>

  <div class="feature-card">
    <div class="feature-icon">&#127922;</div>
    <h3>Advanced Sampling</h3>
    <p>Multiple decoding strategies: Greedy, Temperature, Top-k, and Top-p (nucleus) sampling. All implemented as reusable standalone functions.</p>
    <span class="badge badge-stable">Stable</span>
  </div>
</div>

---

## Quick Start

### Requirements

- **NVIDIA GPU**: Compute Capability 7.0+ (Volta or newer)
- **CUDA Toolkit**: 11.0 or higher
- **CMake**: 3.18 or higher
- **C++ Compiler**: GCC 9+ or Clang 10+

### Installation

```bash
# Clone repository
git clone https://github.com/LessUp/tiny-llm.git
cd tiny-llm

# Build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Run tests
ctest --output-on-failure
```

<div class="cta-box">
  <h4>Ready to dive deeper?</h4>
  <div class="cta-links">
    <a href="{{ site.baseurl }}/docs/en/QUICKSTART" class="btn btn-outline">Read Full Quickstart</a>
    <a href="{{ site.baseurl }}/docs/en/ARCHITECTURE" class="btn btn-outline">Explore Architecture</a>
  </div>
</div>

---

## Documentation

<div class="docs-grid">
  <a href="{{ site.baseurl }}/docs/en/QUICKSTART" class="doc-card">
    <div class="doc-icon">&#128640;</div>
    <h4>Quick Start</h4>
    <p>Get up and running in minutes</p>
  </a>

  <a href="{{ site.baseurl }}/docs/en/ARCHITECTURE" class="doc-card">
    <div class="doc-icon">&#127959;</div>
    <h4>Architecture</h4>
    <p>System design and components</p>
  </a>

  <a href="{{ site.baseurl }}/docs/en/API" class="doc-card">
    <div class="doc-icon">&#128214;</div>
    <h4>API Reference</h4>
    <p>Complete API documentation</p>
  </a>

  <a href="{{ site.baseurl }}/docs/en/DEVELOPER" class="doc-card">
    <div class="doc-icon">&#128295;</div>
    <h4>Developer Guide</h4>
    <p>Development and contribution</p>
  </a>

  <a href="{{ site.baseurl }}/docs/en/BENCHMARKS" class="doc-card">
    <div class="doc-icon">&#9889;</div>
    <h4>Benchmarks</h4>
    <p>Performance metrics and profiling</p>
  </a>

  <a href="{{ site.baseurl }}/docs/en/TROUBLESHOOTING" class="doc-card">
    <div class="doc-icon">&#128269;</div>
    <h4>Troubleshooting</h4>
    <p>Common issues and solutions</p>
  </a>
</div>

---

## Language Support

<div class="language-selector">
  <p>Documentation available in multiple languages:</p>
  <div class="lang-links">
    <a href="{{ site.baseurl }}/docs/en/" class="lang-link active">
      <span class="lang-flag">&#127482;&#127480;</span>
      <span class="lang-name">English</span>
    </a>
    <a href="{{ site.baseurl }}/docs/zh/" class="lang-link">
      <span class="lang-flag">&#127464;&#127475;</span>
      <span class="lang-name">简体中文</span>
    </a>
  </div>
</div>

---

## Performance Highlights

<div class="performance-grid">
  <div class="perf-item">
    <div class="perf-bar">
      <div class="perf-fill" style="width: 50%"></div>
    </div>
    <div class="perf-info">
      <span class="perf-title">Memory Usage</span>
      <span class="perf-value">-50% with W8A16</span>
    </div>
  </div>

  <div class="perf-item">
    <div class="perf-bar">
      <div class="perf-fill" style="width: 80%"></div>
    </div>
    <div class="perf-info">
      <span class="perf-title">Tensor Core Util</span>
      <span class="perf-value">~80% peak</span>
    </div>
  </div>

  <div class="perf-item">
    <div class="perf-bar">
      <div class="perf-fill" style="width: 95%"></div>
    </div>
    <div class="perf-info">
      <span class="perf-title">Test Coverage</span>
      <span class="perf-value">Comprehensive</span>
    </div>
  </div>
</div>

---

## Contributing

We welcome contributions! Check out our [Contributing Guide]({{ site.baseurl }}/CONTRIBUTING) to get started.

<div class="contrib-grid">
  <a href="https://github.com/LessUp/tiny-llm/issues" class="contrib-link">
    <span class="contrib-icon">&#128027;</span>
    <span>Report Issues</span>
  </a>
  <a href="https://github.com/LessUp/tiny-llm/pulls" class="contrib-link">
    <span class="contrib-icon">&#128260;</span>
    <span>Submit PRs</span>
  </a>
  <a href="https://github.com/LessUp/tiny-llm/discussions" class="contrib-link">
    <span class="contrib-icon">&#128172;</span>
    <span>Discussions</span>
  </a>
</div>

---

## License

Distributed under the MIT License. See [LICENSE]({{ site.repo_url }}/blob/master/LICENSE) for more information.

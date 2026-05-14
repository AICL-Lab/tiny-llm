<script setup lang="ts">
// InlineLogo.vue - 内联 SVG logo 组件
// 使用 CSS 变量控制颜色，自动适应深色/浅色模式
</script>

<template>
  <svg
    class="inline-logo"
    viewBox="0 0 200 200"
    xmlns="http://www.w3.org/2000/svg"
    role="img"
    aria-label="Tiny-LLM Logo"
  >
    <defs>
      <linearGradient id="logo-gradient" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" class="stop-primary" />
        <stop offset="100%" class="stop-cuda" />
      </linearGradient>
    </defs>
    <!-- Background circle -->
    <circle cx="100" cy="100" r="95" class="bg-circle" />
    <!-- Geometric "T" shape -->
    <path
      d="M40 35 L160 35 L160 55 L110 55 L110 165 L90 165 L90 55 L40 55 Z"
      class="logo-path"
    />
    <!-- Circuit accent dots -->
    <circle cx="100" cy="35" r="8" class="accent-cuda" />
    <circle cx="100" cy="165" r="8" class="accent-primary" />
    <!-- Side accents -->
    <rect x="30" y="90" width="15" height="4" rx="2" class="accent-cuda-muted" />
    <rect x="155" y="90" width="15" height="4" rx="2" class="accent-primary-muted" />
  </svg>
</template>

<style scoped>
.inline-logo {
  width: 100%;
  height: 100%;
}

/* Gradient stops - controlled by CSS variables */
.stop-primary {
  stop-color: var(--svg-primary);
}

.stop-cuda {
  stop-color: var(--svg-cuda);
}

/* Background circle */
.bg-circle {
  fill: url(#logo-gradient);
  opacity: var(--svg-bg-opacity);
}

/* Main logo path */
.logo-path {
  fill: url(#logo-gradient);
}

/* Accent elements */
.accent-primary {
  fill: var(--svg-primary);
}

.accent-cuda {
  fill: var(--svg-cuda);
}

/* Muted accents */
.accent-primary-muted {
  fill: var(--svg-primary);
  opacity: 0.6;
}

.accent-cuda-muted {
  fill: var(--svg-cuda);
  opacity: 0.6;
}
</style>

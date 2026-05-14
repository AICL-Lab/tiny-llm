<script setup lang="ts">
// Layout.vue - 扩展 VitePress 默认布局
// 使用插槽替换导航栏 logo 为内联 SVG
import DefaultTheme from 'vitepress/theme'
import InlineLogo from './components/InlineLogo.vue'

const { Layout } = DefaultTheme
</script>

<template>
  <Layout>
    <!-- 替换导航栏 logo -->
    <template #nav-bar-title>
      <div class="nav-logo-container">
        <InlineLogo class="nav-logo" />
        <span class="nav-title">Tiny-LLM</span>
      </div>
    </template>
  </Layout>
</template>

<style scoped>
.nav-logo-container {
  display: flex;
  align-items: center;
  gap: 10px;
  text-decoration: none;
}

.nav-logo {
  width: 24px;
  height: 24px;
  flex-shrink: 0;
}

.nav-title {
  font-size: 1rem;
  font-weight: 600;
  color: var(--vp-c-text-1);
  font-family: var(--vp-font-family-mono);
  letter-spacing: -0.02em;
}

/* 悬停效果 */
.nav-logo-container:hover .nav-title {
  color: var(--tiny-llm-primary);
}
</style>

<script setup lang="ts">
defineProps<{
  title: string
  icon?: string
}>()
</script>

<template>
  <div class="tech-spec">
    <div class="spec-header">
      <span v-if="icon" class="icon">{{ icon }}</span>
      <h3>{{ title }}</h3>
    </div>
    <div class="spec-body">
      <slot />
    </div>
  </div>
</template>

<style scoped>
.tech-spec {
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  border-left: 3px solid var(--tiny-llm-primary);
  border-radius: 8px;
  padding: 16px 20px;
  margin: 16px 0;
}

.tech-spec .spec-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
}

.tech-spec .spec-header h3 {
  margin: 0;
  font-size: 1em;
  color: var(--tiny-llm-primary);
  font-family: var(--vp-font-family-base);
}

.tech-spec .spec-header .icon {
  font-size: 1.2em;
}

.tech-spec .spec-body {
  font-size: 0.9em;
  line-height: 1.8;
}

.tech-spec .spec-body :deep(p) {
  margin: 0;
}

.tech-spec .spec-body :deep(ul) {
  margin: 0;
  padding-left: 1.2em;
}

.tech-spec .spec-body :deep(li) {
  margin: 4px 0;
}

.tech-spec .spec-body :deep(code) {
  background: var(--vp-c-bg-mute);
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 0.9em;
}
</style>

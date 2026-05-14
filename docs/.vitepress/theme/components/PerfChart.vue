<script setup lang="ts">
interface Metric {
  label: string
  value: number | string
  pct?: number
  unit?: string
}

defineProps<{
  metrics: Metric[]
  title?: string
}>()
</script>

<template>
  <div class="perf-chart">
    <h4 v-if="title" class="chart-title">{{ title }}</h4>
    <div class="metric-row" v-for="m in metrics" :key="m.label">
      <span class="label">{{ m.label }}</span>
      <div class="bar-container">
        <div
          class="bar"
          :style="{ width: (m.pct ?? 50) + '%' }"
        >
          <span class="value">{{ m.value }}{{ m.unit ? ' ' + m.unit : '' }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.perf-chart {
  margin: 16px 0;
}

.chart-title {
  font-size: 0.9em;
  font-weight: 600;
  color: var(--vp-c-text-1);
  margin: 0 0 12px 0;
}

.metric-row {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 8px;
}

.label {
  min-width: 100px;
  font-size: 0.85em;
  font-family: var(--vp-font-family-mono);
  color: var(--vp-c-text-2);
}

.bar-container {
  flex: 1;
  background: var(--vp-c-bg-mute);
  border-radius: 4px;
  height: 24px;
  overflow: hidden;
}

.bar {
  height: 100%;
  background: linear-gradient(90deg, var(--tiny-llm-primary), var(--tiny-llm-cuda));
  border-radius: 4px;
  display: flex;
  align-items: center;
  justify-content: flex-end;
  padding-right: 10px;
  min-width: fit-content;
  transition: width 0.6s ease;
}

.value {
  font-size: 0.75em;
  font-family: var(--vp-font-family-mono);
  color: #fff;
  font-weight: 500;
}
</style>

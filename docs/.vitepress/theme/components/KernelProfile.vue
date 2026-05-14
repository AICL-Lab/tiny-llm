<script setup lang="ts">
defineProps<{
  name: string
  tensorCore?: string | number
  memBw?: string | number
  occupancy?: string | number
  throughput?: string | number
}>()
</script>

<template>
  <div class="kernel-profile">
    <code class="kernel-name">{{ name }}</code>
    <div class="metrics">
      <div v-if="tensorCore !== undefined" class="metric">
        <span class="value">{{ tensorCore }}<span class="unit">%</span></span>
        <span class="label">Tensor Core</span>
      </div>
      <div v-if="memBw !== undefined" class="metric">
        <span class="value">{{ memBw }}<span class="unit">GB/s</span></span>
        <span class="label">Memory BW</span>
      </div>
      <div v-if="occupancy !== undefined" class="metric">
        <span class="value">{{ occupancy }}<span class="unit">%</span></span>
        <span class="label">Occupancy</span>
      </div>
      <div v-if="throughput !== undefined" class="metric">
        <span class="value">{{ throughput }}<span class="unit">%</span></span>
        <span class="label">Throughput</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.kernel-profile {
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 16px 20px;
  margin: 16px 0;
}

.kernel-name {
  font-family: var(--vp-font-family-mono);
  font-size: 0.9em;
  color: var(--tiny-llm-cuda);
  background: oklch(0.70 0.08 95 / 0.14);
  padding: 4px 10px;
  border-radius: 4px;
}

.metrics {
  display: flex;
  gap: 32px;
  margin-top: 14px;
}

.metric {
  text-align: center;
}

.metric .value {
  font-family: var(--vp-font-family-mono);
  font-size: 1.5em;
  font-weight: 600;
  color: var(--tiny-llm-primary);
}

.metric .unit {
  font-size: 0.5em;
  color: var(--vp-c-text-3);
  margin-left: 2px;
}

.metric .label {
  display: block;
  font-size: 0.75em;
  color: var(--vp-c-text-2);
  margin-top: 4px;
  font-family: var(--vp-font-family-base);
}

@media (max-width: 640px) {
  .metrics {
    flex-wrap: wrap;
    gap: 20px;
  }

  .metric {
    flex: 1 1 40%;
  }
}
</style>

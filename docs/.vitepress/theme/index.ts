import DefaultTheme from 'vitepress/theme'
import './custom.css'

// Custom Vue components
import TechSpec from './components/TechSpec.vue'
import PerfChart from './components/PerfChart.vue'
import KernelProfile from './components/KernelProfile.vue'

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    // Register global components
    app.component('TechSpec', TechSpec)
    app.component('PerfChart', PerfChart)
    app.component('KernelProfile', KernelProfile)
  }
}

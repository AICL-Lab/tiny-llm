import DefaultTheme from 'vitepress/theme'
import './custom.css'

// Custom layout (extends default with inline SVG logo)
import Layout from './Layout.vue'

// Custom Vue components
import TechSpec from './components/TechSpec.vue'
import PerfChart from './components/PerfChart.vue'
import KernelProfile from './components/KernelProfile.vue'
import InlineLogo from './components/InlineLogo.vue'

export default {
  extends: DefaultTheme,
  Layout,
  enhanceApp({ app }) {
    // Register global components
    app.component('TechSpec', TechSpec)
    app.component('PerfChart', PerfChart)
    app.component('KernelProfile', KernelProfile)
    app.component('InlineLogo', InlineLogo)
  }
}

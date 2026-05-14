import DefaultTheme from 'vitepress/theme'
import { onBeforeMount } from 'vue'
import { useRoute } from 'vitepress'
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
  },
  setup() {
    const route = useRoute()

    onBeforeMount(() => {
      // Only perform language detection on the root page
      const path = route.path.replace(/\/$/, '').replace(/\/index\.html$/, '')

      if (path === '' || path === '/tiny-llm') {
        const storedLang = localStorage.getItem('tiny-llm-lang')

        // If user has no stored preference, try to detect from browser
        if (!storedLang) {
          const browserLang = navigator.language.toLowerCase()
          const targetLang = browserLang.startsWith('zh') ? 'zh' : 'en'

          // Store the preference
          localStorage.setItem('tiny-llm-lang', targetLang)

          // Redirect to the appropriate language version
          const basePath = import.meta.env.BASE_URL || '/'
          const targetPath = `${basePath}${targetLang}/`
          window.location.replace(targetPath)
        }
      }
    })
  }
}

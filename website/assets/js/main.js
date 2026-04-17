// ═══════════════════════════════════════════════════════════════
// Tiny-LLM Main JavaScript
// ═══════════════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', () => {
  initBackToTop();
  initMobileMenu();
  initThemeToggle();
});

// ─── Back to Top Button ──────────────────────────────────────
function initBackToTop() {
  const button = document.querySelector('.back-to-top');
  if (!button) return;

  const toggleVisibility = () => {
    if (window.scrollY > 300) {
      button.classList.add('visible');
    } else {
      button.classList.remove('visible');
    }
  };

  window.addEventListener('scroll', toggleVisibility);
  button.addEventListener('click', () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });
}

// ─── Mobile Menu Toggle ──────────────────────────────────────
function initMobileMenu() {
  const menuToggle = document.querySelector('.menu-toggle');
  const sideBar = document.querySelector('.side-bar');
  if (!menuToggle || !sideBar) return;

  menuToggle.addEventListener('click', () => {
    sideBar.classList.toggle('open');
    menuToggle.classList.toggle('active');
  });

  // Close menu when clicking outside
  document.addEventListener('click', (e) => {
    if (!sideBar.contains(e.target) && !menuToggle.contains(e.target)) {
      sideBar.classList.remove('open');
      menuToggle.classList.remove('active');
    }
  });
}

// ─── Theme Toggle ────────────────────────────────────────────
function initThemeToggle() {
  // Create theme toggle button if not exists
  const headerActions = document.querySelector('.header-actions');
  if (!headerActions) return;

  const toggle = document.createElement('button');
  toggle.className = 'theme-toggle';
  toggle.setAttribute('aria-label', 'Toggle theme');
  toggle.innerHTML = getThemeIcon();
  headerActions.prepend(toggle);

  toggle.addEventListener('click', () => {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    toggle.innerHTML = getThemeIcon(newTheme);
  });
}

function getThemeIcon(theme = 'dark') {
  if (theme === 'dark') {
    return '<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="5"/><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/></svg>';
  }
  return '<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>';
}

// ─── Service Worker Update Notification ──────────────────────
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.addEventListener('controllerchange', () => {
    console.log('[SW] New version available, reloading...');
    window.location.reload();
  });
}

// ─── Copy Code Button ───────────────────────────────────────
function initCopyCodeButtons() {
  const codeBlocks = document.querySelectorAll('pre');
  codeBlocks.forEach((block) => {
    const button = document.createElement('button');
    button.className = 'copy-code-btn';
    button.textContent = 'Copy';
    button.addEventListener('click', async () => {
      const code = block.querySelector('code')?.textContent;
      if (!code) return;
      await navigator.clipboard.writeText(code);
      button.textContent = 'Copied!';
      setTimeout(() => { button.textContent = 'Copy'; }, 2000);
    });
    block.style.position = 'relative';
    block.appendChild(button);
  });
}

document.addEventListener('DOMContentLoaded', initCopyCodeButtons);

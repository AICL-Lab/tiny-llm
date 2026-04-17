// ═══════════════════════════════════════════════════════════════
// Tiny-LLM Service Worker - PWA Support
// Strategy: Cache-First with Network Fallback
// Version: 2.1.0
// ═══════════════════════════════════════════════════════════════

const CACHE_VERSION = 'v2.1.0';
const CACHE_NAME = `tiny-llm-${CACHE_VERSION}`;
const STATIC_CACHE = `${CACHE_NAME}-static`;
const DOCS_CACHE = `${CACHE_NAME}-docs`;

// Assets to pre-cache on install
const PRECACHE_URLS = [
  '/tiny-llm/',
  '/tiny-llm/index.html',
  '/tiny-llm/manifest.json',
  '/tiny-llm/assets/css/custom.css',
  '/tiny-llm/assets/js/main.js',
  '/tiny-llm/404.html'
];

// ═══════════════════════════════════════════════════════════════
// INSTALL EVENT - Pre-cache critical assets
// ═══════════════════════════════════════════════════════════════
self.addEventListener('install', (event) => {
  console.log('[SW] Installing v' + CACHE_VERSION);

  event.waitUntil(
    caches.open(STATIC_CACHE)
      .then((cache) => {
        console.log('[SW] Pre-caching assets');
        return cache.addAll(PRECACHE_URLS);
      })
      .then(() => self.skipWaiting())
      .catch((err) => console.error('[SW] Pre-cache failed:', err))
  );
});

// ═══════════════════════════════════════════════════════════════
// ACTIVATE EVENT - Clean up old caches
// ═══════════════════════════════════════════════════════════════
self.addEventListener('activate', (event) => {
  console.log('[SW] Activating v' + CACHE_VERSION);

  event.waitUntil(
    caches.keys()
      .then((cacheNames) => {
        return Promise.all(
          cacheNames
            .filter((name) => !name.startsWith(CACHE_NAME))
            .map((name) => caches.delete(name))
        );
      })
      .then(() => self.clients.claim())
  );
});

// ═══════════════════════════════════════════════════════════════
// FETCH EVENT - Cache strategies
// ═══════════════════════════════════════════════════════════════
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // Skip non-GET requests
  if (request.method !== 'GET') return;

  // Skip external requests
  if (!url.pathname.startsWith('/tiny-llm/')) return;

  // Choose strategy based on request type
  if (isHTML(request)) {
    event.respondWith(networkFirst(request));
  } else if (isCSS(request) || isJS(request)) {
    event.respondWith(staleWhileRevalidate(request));
  } else if (isImage(request)) {
    event.respondWith(cacheFirst(request, 'images'));
  } else {
    event.respondWith(cacheFirst(request));
  }
});

// ═══════════════════════════════════════════════════════════════
// CACHE STRATEGIES
// ═══════════════════════════════════════════════════════════════

// Cache First - For images and static assets
async function cacheFirst(request, cacheName = STATIC_CACHE) {
  const cached = await caches.match(request);
  if (cached) return cached;

  try {
    const response = await fetch(request);
    if (response.ok) {
      const cache = await caches.open(cacheName);
      cache.put(request, response.clone());
    }
    return response;
  } catch (err) {
    console.error('[SW] Cache-first fetch failed:', err);
    return new Response('Offline', { status: 503 });
  }
}

// Network First - For HTML pages (fresh content priority)
async function networkFirst(request) {
  const cache = await caches.open(DOCS_CACHE);

  try {
    const response = await fetch(request);
    if (response.ok) {
      cache.put(request, response.clone());
    }
    return response;
  } catch (err) {
    console.log('[SW] Network failed, serving cached:', request.url);
    return cache.match(request) || caches.match('/tiny-llm/404.html');
  }
}

// Stale While Revalidate - For CSS/JS (fast load + background update)
async function staleWhileRevalidate(request) {
  const cache = await caches.open(STATIC_CACHE);
  const cached = await cache.match(request);

  const fetchPromise = fetch(request)
    .then((response) => {
      if (response.ok) {
        cache.put(request, response.clone());
      }
      return response;
    })
    .catch(() => cached);

  return cached || fetchPromise;
}

// ═══════════════════════════════════════════════════════════════
// UTILITY FUNCTIONS
// ═══════════════════════════════════════════════════════════════
function isHTML(request) {
  return request.headers.get('accept')?.includes('text/html') ||
         request.url.endsWith('.html') ||
         request.url.endsWith('/');
}

function isCSS(request) {
  return request.url.endsWith('.css') ||
         request.headers.get('accept')?.includes('text/css');
}

function isJS(request) {
  return request.url.endsWith('.js') ||
         request.headers.get('accept')?.includes('javascript');
}

function isImage(request) {
  return request.destination === 'image' ||
         /\.(jpg|jpeg|png|gif|webp|svg|ico)$/i.test(request.url);
}

// Message handler
self.addEventListener('message', (event) => {
  if (event.data?.type === 'SKIP_WAITING') {
    self.skipWaiting();
  }
});

console.log('[SW] Tiny-LLM Service Worker loaded');

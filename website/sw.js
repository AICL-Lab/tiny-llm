// ═══════════════════════════════════════════════════════════════
// Tiny-LLM Service Worker - PWA Support
// Strategy: Cache-First with Network Fallback
// Version: 2.1.0
// ═══════════════════════════════════════════════════════════════

const CACHE_VERSION = 'v2.1.0';
const STATIC_CACHE = `${CACHE_VERSION}-static`;
const DOCS_CACHE = `${CACHE_VERSION}-docs`;
const IMAGE_CACHE = `${CACHE_VERSION}-images`;

// Assets to pre-cache on install
const PRECACHE_ASSETS = [
  '/tiny-llm/',
  '/tiny-llm/index.html',
  '/tiny-llm/manifest.json',
  '/tiny-llm/assets/css/custom.css',
  '/tiny-llm/404.html'
];

// Install Event - Pre-cache critical assets
self.addEventListener('install', (event) => {
  console.log('[SW] Installing...');
  
  event.waitUntil(
    caches.open(STATIC_CACHE)
      .then((cache) => {
        console.log('[SW] Pre-caching assets');
        return cache.addAll(PRECACHE_ASSETS);
      })
      .then(() => self.skipWaiting())
      .catch((error) => {
        console.error('[SW] Pre-cache failed:', error);
      })
  );
});

// Activate Event - Clean up old caches
self.addEventListener('activate', (event) => {
  console.log('[SW] Activating...');
  
  const allowedCaches = [STATIC_CACHE, DOCS_CACHE, IMAGE_CACHE];
  
  event.waitUntil(
    caches.keys()
      .then((cacheNames) => {
        return Promise.all(
          cacheNames.map((cacheName) => {
            if (!allowedCaches.includes(cacheName)) {
              console.log('[SW] Deleting old cache:', cacheName);
              return caches.delete(cacheName);
            }
          })
        );
      })
      .then(() => self.clients.claim())
  );
});

// Fetch Event - Cache strategies
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);
  
  // Skip non-GET requests
  if (request.method !== 'GET') return;
  
  // Skip external requests
  if (!url.pathname.startsWith('/tiny-llm/')) return;
  
  // Strategy based on file type
  if (isHTML(request)) {
    event.respondWith(networkFirstStrategy(request));
  } else if (isCSS(request) || isJS(request)) {
    event.respondWith(staleWhileRevalidateStrategy(request));
  } else if (isImage(request)) {
    event.respondWith(cacheFirstStrategy(request, IMAGE_CACHE));
  } else {
    event.respondWith(cacheFirstStrategy(request));
  }
});

// ═══════════════════════════════════════════════════════════════
// CACHE STRATEGIES
// ═══════════════════════════════════════════════════════════════

// Cache First - For images and static assets
async function cacheFirstStrategy(request, cacheName = STATIC_CACHE) {
  const cachedResponse = await caches.match(request);
  
  if (cachedResponse) {
    return cachedResponse;
  }
  
  try {
    const networkResponse = await fetch(request);
    if (networkResponse.ok) {
      const cache = await caches.open(cacheName);
      cache.put(request, networkResponse.clone());
    }
    return networkResponse;
  } catch (error) {
    console.error('[SW] Cache-first fetch failed:', error);
    return new Response('Offline - Resource not cached', {
      status: 503,
      statusText: 'Service Unavailable'
    });
  }
}

// Network First - For HTML pages (fresh content priority)
async function networkFirstStrategy(request) {
  const cache = await caches.open(DOCS_CACHE);
  
  try {
    const networkResponse = await fetch(request);
    if (networkResponse.ok) {
      cache.put(request, networkResponse.clone());
    }
    return networkResponse;
  } catch (error) {
    console.log('[SW] Network failed, serving cached:', request.url);
    const cachedResponse = await cache.match(request);
    
    if (cachedResponse) {
      return cachedResponse;
    }
    
    // Return custom offline page
    return caches.match('/tiny-llm/404.html');
  }
}

// Stale While Revalidate - For CSS/JS (fast load + background update)
async function staleWhileRevalidateStrategy(request) {
  const cache = await caches.open(STATIC_CACHE);
  const cachedResponse = await cache.match(request);
  
  const fetchPromise = fetch(request)
    .then((networkResponse) => {
      if (networkResponse.ok) {
        cache.put(request, networkResponse.clone());
        // Notify clients about update
        notifyClientsOfUpdate(request.url);
      }
      return networkResponse;
    })
    .catch(() => cachedResponse);
  
  return cachedResponse || fetchPromise;
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

// Notify clients about content updates
async function notifyClientsOfUpdate(url) {
  const clients = await self.clients.matchAll({ type: 'window' });
  clients.forEach((client) => {
    client.postMessage({
      type: 'UPDATE_AVAILABLE',
      url: url,
      version: CACHE_VERSION
    });
  });
}

// Handle messages from clients
self.addEventListener('message', (event) => {
  if (event.data.type === 'SKIP_WAITING') {
    self.skipWaiting();
  }
  
  if (event.data.type === 'GET_VERSION') {
    event.ports[0].postMessage({ version: CACHE_VERSION });
  }
});

// Background sync for offline form submissions (if needed later)
self.addEventListener('sync', (event) => {
  if (event.tag === 'sync-search') {
    event.waitUntil(handleBackgroundSync());
  }
});

async function handleBackgroundSync() {
  // Implementation for background sync
  console.log('[SW] Background sync executed');
}

// Push notifications (if enabled later)
self.addEventListener('push', (event) => {
  const options = {
    body: event.data?.text() || 'Tiny-LLM update available',
    icon: '/tiny-llm/assets/icons/icon-192x192.png',
    badge: '/tiny-llm/assets/icons/icon-72x72.png',
    tag: 'tiny-llm-update',
    requireInteraction: false,
    data: {
      url: '/tiny-llm/changelog/'
    }
  };
  
  event.waitUntil(
    self.registration.showNotification('Tiny-LLM Documentation', options)
  );
});

// Notification click handler
self.addEventListener('notificationclick', (event) => {
  event.notification.close();
  
  event.waitUntil(
    clients.openWindow(event.notification.data?.url || '/tiny-llm/')
  );
});

console.log('[SW] Tiny-LLM Service Worker loaded');

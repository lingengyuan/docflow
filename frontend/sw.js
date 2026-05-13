const DOCFLOW_CACHE = 'docflow-shell-v1';
const DOCFLOW_SHELL = [
  '/',
  '/styles.css',
  '/favicon.svg',
  '/js/state.js',
  '/js/icons.js',
  '/js/shared-ui.js',
  '/js/i18n.js',
  '/js/app-shell.js',
];

self.addEventListener('install', event => {
  event.waitUntil(caches.open(DOCFLOW_CACHE).then(cache => cache.addAll(DOCFLOW_SHELL)));
  self.skipWaiting();
});

self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys => Promise.all(
      keys.filter(key => key !== DOCFLOW_CACHE).map(key => caches.delete(key))
    ))
  );
  self.clients.claim();
});

self.addEventListener('fetch', event => {
  const request = event.request;
  const url = new URL(request.url);
  if (
    request.method !== 'GET'
    || url.origin !== self.location.origin
    || url.pathname.startsWith('/api/')
  ) {
    return;
  }
  event.respondWith(
    fetch(request)
      .then(response => {
        const copy = response.clone();
        caches.open(DOCFLOW_CACHE).then(cache => cache.put(request, copy));
        return response;
      })
      .catch(() => caches.match(request))
  );
});

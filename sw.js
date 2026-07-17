const CACHE = "mrakoradar-v1";
const SHELL = ["./", "./index.html", "./manifest.json", "./icon-192.png", "./icon-512.png"];

self.addEventListener("install", e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(SHELL)));
  self.skipWaiting();
});
self.addEventListener("activate", e => {
  e.waitUntil(
    caches.keys().then(keys => Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k))))
  );
  self.clients.claim();
});
// appku (shell) servírovat z keše, radarová data vždy ze sítě (musí být čerstvá)
self.addEventListener("fetch", e => {
  const url = new URL(e.request.url);
  if (url.origin !== location.origin) return;   // dlaždice/API RainViewer necachovat
  e.respondWith(
    caches.match(e.request).then(cached => cached || fetch(e.request))
  );
});

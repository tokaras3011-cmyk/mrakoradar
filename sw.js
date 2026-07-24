const CACHE = "mrakoradar-v3";
const SHELL = [
  "./", "./index.html", "./manifest.json", "./icon-192.png", "./icon-512.png",
  "./model/model.json", "./model/weights.bin", "./model/labels.json",
];

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
self.addEventListener("fetch", e => {
  const url = new URL(e.request.url);
  if (url.origin !== location.origin) return;   // dlaždice/API RainViewer necachovat

  // appku samotnou (HTML) vždy nejdřív zkusit ze sítě, ať se po nasazení nové verze
  // hned projeví — do keše padá jen jako záloha pro offline použití
  const isAppShell = e.request.mode === "navigate" || url.pathname.endsWith("index.html") || url.pathname.endsWith("/");
  if (isAppShell) {
    e.respondWith(
      fetch(e.request).then(res => {
        caches.open(CACHE).then(c => c.put(e.request, res.clone()));
        return res;
      }).catch(() => caches.match(e.request))
    );
    return;
  }
  // statické assety (ikony, model) klidně z keše, mění se zřídka
  e.respondWith(caches.match(e.request).then(cached => cached || fetch(e.request)));
});

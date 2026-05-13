// ── PWA shell registration ──
if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/sw.js').catch(() => {
      // PWA support is optional; the app still works as a normal local page.
    });
  });
}

// ── PWA shell registration ──
if ('serviceWorker' in navigator) {
  const registerDocFlowServiceWorker = () => {
    navigator.serviceWorker.register('/sw.js').catch(() => {
      // PWA support is optional; the app still works as a normal local page.
    });
  };
  if (document.readyState === 'complete') {
    registerDocFlowServiceWorker();
  } else {
    window.addEventListener('load', registerDocFlowServiceWorker, { once: true });
  }
}

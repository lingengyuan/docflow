const DOCFLOW_SCRIPT_ORDER = [
  '/js/state.js',
  '/js/icons.js',
  '/js/shared-ui.js',
  '/js/i18n.js',
  '/js/app-shell.js',
  '/js/settings.js',
  '/js/settings-models.js',
  '/js/settings-data.js',
  '/js/chat.js',
  '/js/notes.js',
  '/js/chat-stream.js',
  '/js/generated/stream-parser.js',
  '/js/chat-actions.js',
  '/js/source-preview.js',
  '/js/source-preview-actions.js',
  '/js/library.js',
  '/js/library-render.js',
  '/js/library-knowledge.js',
  '/js/library-actions.js',
  '/js/history.js',
  '/js/queue-upload.js',
  '/js/pwa.js',
  '/js/settings-bootstrap.js',
];

function loadClassicScript(src) {
  return new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = src;
    script.onload = resolve;
    script.onerror = () => reject(new Error(`Failed to load ${src}`));
    document.body.appendChild(script);
  });
}

async function bootDocFlowShell() {
  const root = document.getElementById('docflow-shell-root');
  const partial = root?.dataset.partial || '/partials/app.html';
  try {
    const response = await fetch(partial);
    if (!response.ok) throw new Error(`Failed to load ${partial}`);
    root.innerHTML = await response.text();
    for (const script of DOCFLOW_SCRIPT_ORDER) {
      await loadClassicScript(script);
    }
    window.applyI18n?.();
  } catch (error) {
    root.innerHTML = `<div class="m-auto max-w-sm rounded-lg bg-white px-4 py-3 text-sm font-semibold text-error shadow-sm">DocFlow 打开失败，请刷新页面重试。</div>`;
    console.error(error);
  }
}

bootDocFlowShell();

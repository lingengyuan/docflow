function showConfirmDialog({ title, message, confirmText = '确认' }) {
  const modal = document.getElementById('confirm-modal');
  document.getElementById('confirm-title').textContent = title;
  document.getElementById('confirm-message').textContent = message;
  document.getElementById('confirm-action').textContent = confirmText;
  modal.classList.remove('hidden');
  modal.classList.add('flex');
  return new Promise(resolve => {
    pendingConfirmResolve = resolve;
  });
}

function closeConfirmDialog(confirmed) {
  const modal = document.getElementById('confirm-modal');
  modal.classList.add('hidden');
  modal.classList.remove('flex');
  const resolve = pendingConfirmResolve;
  pendingConfirmResolve = null;
  if (resolve) resolve(Boolean(confirmed));
}

document.addEventListener('keydown', event => {
  if (event.key === 'Escape' && pendingConfirmResolve) {
    closeConfirmDialog(false);
  }
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
    event.preventDefault();
    document.getElementById('global-search-input')?.focus();
  }
});

function handleGlobalSearchKey(event) {
  if (event.key !== 'Enter') return;
  const value = event.currentTarget.value.trim();
  if (!value) return;
  switchView('chat');
  const input = document.getElementById('input');
  input.value = value;
  autoResize(input);
  event.currentTarget.value = '';
  input.focus();
}

// ── View switching ──
function switchView(view) {
  document.querySelectorAll('.view').forEach(v => v.classList.add('hidden'));
  const targetView = document.getElementById(`view-${view}`);
  targetView.classList.remove('hidden');
  targetView.focus({ preventScroll: true });

  document.querySelectorAll('.nav-btn').forEach(b => {
    b.classList.remove('active-nav');
    b.classList.add('opacity-80');
    b.removeAttribute('aria-current');
  });
  const btn = document.getElementById(`nav-${view}`) || (view === 'history' ? document.getElementById('nav-settings') : null);
  if (btn) {
    btn.classList.add('active-nav');
    btn.classList.remove('opacity-80');
    btn.setAttribute('aria-current', 'page');
  }

  if (view === 'chat') { loadConversations(); loadQueryScopeOptions(); loadLatestAnswerPreview(); }
  if (view === 'library') { refreshFiles(); pollQueueOnce(); }
  if (view === 'source') loadDefaultSourcePreview();
  if (view === 'notes') refreshNotesView();
  if (view === 'settings') refreshSettings();
  if (view === 'history') refreshHistory();
}

// Init active nav style
document.getElementById('nav-chat').classList.add('active-nav');

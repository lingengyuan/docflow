// ── Conversations ──
document.getElementById('conversation-menu').addEventListener('click', e => e.stopPropagation());
document.getElementById('health-panel').addEventListener('click', e => e.stopPropagation());

function toggleConversationMenu(e) {
  e.stopPropagation();
  loadConversations();
  document.getElementById('conversation-menu').classList.toggle('hidden');
}

async function loadConversations() {
  try {
    const items = await fetch(`${API}/api/conversations`).then(r => r.json());
    conversationItems = items;
    renderConversationList();
    updateConversationLabel();
  } catch {
    document.getElementById('conversation-list').innerHTML =
      '<div class="px-3 py-8 text-center text-xs text-error">对话加载失败</div>';
  }
}

function renderConversationList() {
  const list = document.getElementById('conversation-list');
  if (!conversationItems.length) {
    list.innerHTML = '<div class="px-3 py-8 text-center text-xs text-on-surface-variant/60">暂无对话</div>';
    return;
  }
  list.innerHTML = conversationItems.map(item => {
    const active = item.id === currentConversationId;
    const title = escHtml(item.title || '未命名对话');
    return `<div class="group flex items-start gap-2 rounded-lg px-3 py-2 cursor-pointer ${active ? 'bg-primary-container text-on-primary-container' : 'hover:bg-surface-container'}"
        role="button" tabindex="0" aria-label="打开对话：${title}"
        onclick="switchConversation(${item.id})"
        onkeydown="handleConversationKey(event, ${item.id})">
      <span class="material-symbols-outlined ${active ? 'icon-fill text-primary' : 'text-on-surface-variant/50'}" style="font-size:17px">chat_bubble</span>
      <div class="min-w-0 flex-1">
        <div class="text-xs font-semibold line-clamp-1">${title}</div>
        <div class="text-[11px] text-on-surface-variant/60 line-clamp-1">${escHtml(item.last_message || '空对话')}</div>
        <div class="text-[10px] text-on-surface-variant/45 mt-1">${item.message_count || 0} 条 · ${(item.updated_at || '').slice(0,16)}</div>
      </div>
      <button onclick="deleteConversation(event, ${item.id})" class="opacity-70 group-hover:opacity-100 w-7 h-7 flex items-center justify-center rounded-lg bg-error/5 text-error hover:bg-error hover:text-on-error transition-all" title="删除对话" aria-label="删除对话">
        <span class="material-symbols-outlined" style="font-size:15px">delete</span>
      </button>
    </div>`;
  }).join('');
}

function updateConversationLabel() {
  const current = conversationItems.find(item => item.id === currentConversationId);
  document.getElementById('conversation-current').textContent = current?.title || '新对话';
}

async function createNewConversation() {
  const item = await fetch(`${API}/api/conversations`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ title: '' }),
  }).then(r => r.json());
  currentConversationId = item.id;
  resetChatMessages();
  await loadConversations();
  document.getElementById('conversation-menu').classList.add('hidden');
  document.getElementById('input').focus();
}

async function switchConversation(id) {
  currentConversationId = id;
  await loadConversationMessages(id);
  updateConversationLabel();
  document.getElementById('conversation-menu').classList.add('hidden');
  switchView('chat');
}

async function deleteConversation(event, id) {
  event.stopPropagation();
  const item = conversationItems.find(conversation => conversation.id === id);
  const title = item?.title || '未命名对话';
  const confirmed = await showConfirmDialog({
    title: '删除这个对话？',
    message: `这会删除“${title}”以及其中的消息。`,
    confirmText: '删除对话',
  });
  if (!confirmed) return;
  const r = await fetch(`${API}/api/conversations/${id}`, { method: 'DELETE' });
  if (!r.ok) return;
  if (currentConversationId === id) {
    currentConversationId = null;
    resetChatMessages();
  }
  await loadConversations();
}

async function loadConversationMessages(id) {
  const messages = await fetch(`${API}/api/conversations/${id}/messages`).then(r => r.json());
  renderConversationMessages(messages);
}

function resetChatMessages() {
  const inner = document.querySelector('#messages .max-w-2xl');
  inner.dataset.latestPreview = '';
  inner.innerHTML = welcomeHtml();
  document.getElementById('messages').scrollTop = 0;
  lastCitations = [];
  lastRelatedNotes = [];
  chatPreviewCitation = null;
  renderChatContextSources([]);
  renderRelatedNotes([]);
  renderChatSourcePreview(null);
  updateConversationLabel();
}

function welcomeHtml() {
  return `<div class="flex flex-col gap-4">
    <div class="flex items-center gap-2">
      <div class="w-6 h-6 rounded-lg bg-surface-container-highest flex items-center justify-center">
        <span class="material-symbols-outlined text-primary icon-fill" style="font-size:14px">auto_awesome</span>
      </div>
      <span class="text-xs font-bold uppercase tracking-widest text-on-surface-variant">DocFlow</span>
    </div>
    <div class="text-sm text-on-surface leading-relaxed">
      你好，我是 DocFlow — 你的本地知识助手。<br>
      把文件拖进资料库，或去笔记页导入网页、写临时笔记，再回到这里选择范围提问。
      <div class="mt-4 grid grid-cols-1 sm:grid-cols-3 gap-2 text-xs">
        <button onclick="switchView('library')" class="text-left rounded-lg bg-surface-container-low px-3 py-2 hover:bg-surface-container transition-all active:scale-95">
          <span class="block font-semibold text-on-surface">整理文件库</span>
          <span class="text-on-surface-variant/70">上传、打标签、分集合</span>
        </button>
        <button onclick="switchView('notes')" class="text-left rounded-lg bg-surface-container-low px-3 py-2 hover:bg-surface-container transition-all active:scale-95">
          <span class="block font-semibold text-on-surface">采集新资料</span>
          <span class="text-on-surface-variant/70">网页和临时笔记</span>
        </button>
        <button onclick="switchView('settings')" class="text-left rounded-lg bg-surface-container-low px-3 py-2 hover:bg-surface-container transition-all active:scale-95">
          <span class="block font-semibold text-on-surface">检查状态</span>
          <span class="text-on-surface-variant/70">模型、路径、状态</span>
        </button>
      </div>
    </div>
  </div>`;
}

function renderConversationMessages(messages) {
  const msgs = document.getElementById('messages');
  const inner = msgs.querySelector('.max-w-2xl');
  inner.innerHTML = '';
  if (!messages.length) {
    inner.innerHTML = welcomeHtml();
    if (!currentConversationId) loadLatestAnswerPreview();
    return;
  }
  inner.dataset.latestPreview = '';
  messages.forEach(message => {
    if (message.role === 'user') appendUserMessage(message.content, false);
    if (message.role === 'assistant') appendAssistantMessage(message.content, message.citations || [], {
      createdAt: message.created_at,
    });
  });
  msgs.scrollTop = msgs.scrollHeight;
}

async function loadLatestAnswerPreview() {
  if (currentConversationId !== null) return;
  const view = document.getElementById('view-chat');
  const inner = document.querySelector('#messages .max-w-2xl');
  if (!view || view.classList.contains('hidden') || !inner || inner.dataset.latestPreview === 'loaded') return;
  try {
    const items = await fetch(`${API}/api/history?limit=1`).then(r => r.json());
    const item = Array.isArray(items) ? items[0] : null;
    if (!item?.question || !item?.answer) return;
    renderLatestAnswerPreview(item);
  } catch {}
}

function renderLatestAnswerPreview(item) {
  const msgs = document.getElementById('messages');
  const inner = msgs.querySelector('.max-w-2xl');
  const citations = item.citations || [];
  const answer = item.answer || '';
  const createdAt = (item.created_at || '').slice(0, 16);
  inner.dataset.latestPreview = 'loaded';
  inner.innerHTML = `
    <div class="flex flex-col items-end gap-2">
      <div class="max-w-[86%] bg-primary text-on-primary px-5 py-3 rounded-2xl rounded-tr-sm shadow-sm">
        <p class="text-sm leading-relaxed">${escHtml(item.question)}</p>
      </div>
      <span class="text-[10px] font-bold text-on-surface-variant/40 tracking-wider">${escHtml(createdAt || '最近提问')}</span>
    </div>
    <div class="flex flex-col gap-4">
      <div class="flex items-center gap-2">
        <div class="w-8 h-8 rounded-full bg-primary text-on-primary flex items-center justify-center shadow-sm">
          <span class="material-symbols-outlined icon-fill" style="font-size:16px">auto_awesome</span>
        </div>
        <div>
          <span class="block text-sm font-bold text-on-surface">DocFlow</span>
          <span class="block text-[11px] text-on-surface-variant/60">${escHtml(createdAt || '最近回答')}</span>
        </div>
      </div>
      <div class="answer-card px-5 py-4">
        <div class="text-sm font-bold text-on-surface">回答摘要</div>
        <div class="prose mt-3 text-sm text-on-surface max-w-none">${renderMarkdown(answer)}</div>
      </div>
      <div class="flex flex-wrap gap-2">${citationMarkup(citations)}</div>
      <div class="flex flex-wrap items-center justify-between gap-3">
        <div class="flex flex-wrap gap-2">
          <button onclick="copyTextFromButton(this)" data-copy-text="${escHtml(answer)}" class="answer-action" title="复制答案" aria-label="复制答案">
            <span class="material-symbols-outlined" style="font-size:15px">content_copy</span>复制
          </button>
          <button onclick="saveAnswerFromButton(this)" data-answer-text="${escHtml(answer)}" class="answer-action" title="保存为笔记" aria-label="保存为笔记">
            <span class="material-symbols-outlined" style="font-size:15px">note_add</span>保存为笔记
          </button>
          <button onclick="exportAnswerFromButton(this)" data-answer-text="${escHtml(answer)}" class="answer-action" title="导出 Markdown" aria-label="导出 Markdown">
            <span class="material-symbols-outlined" style="font-size:15px">download</span>导出
          </button>
          <button onclick="switchView('notes')" class="answer-action" title="生成知识产物" aria-label="生成知识产物">
            <span class="material-symbols-outlined" style="font-size:15px">lightbulb</span>生成知识产物
          </button>
        </div>
        <span class="text-[11px] text-on-surface-variant/60">${citations.length ? `基于 ${citations.length} 个来源` : '暂无引用来源'}</span>
      </div>
    </div>`;
  lastCitations = citations;
  lastRelatedNotes = [];
  renderChatContextSources(citations);
  renderRelatedNotes([]);
  renderLocalIcons(inner);
  msgs.scrollTop = 0;
}

// ── Query scope controls ──
async function loadQueryScopeOptions(force = false) {
  const status = document.getElementById('query-scope-status');
  const refresh = document.getElementById('query-scope-refresh');
  if (refresh) refresh.disabled = true;
  if (status && force) status.textContent = '刷新中…';
  try {
    const [meta, files] = await Promise.all([
      fetch(`${API}/api/library/meta`).then(r => r.json()).catch(() => libraryMeta),
      fetch(`${API}/api/files?status=done`).then(r => r.json()).catch(() => queryScopeFiles),
    ]);
    libraryMeta = meta || libraryMeta;
    updateSidebarStorageSummary();
    queryScopeFiles = Array.isArray(files) ? files : [];
    renderQueryScopeOptions();
    if (status) status.textContent = queryScopeLabel();
  } catch (e) {
    if (status) status.textContent = '范围读取失败';
  } finally {
    if (refresh) refresh.disabled = false;
  }
}

function renderQueryScopeOptions() {
  const collectionSelect = document.getElementById('query-scope-collection');
  const fileSelect = document.getElementById('query-scope-file');
  if (collectionSelect) {
    const value = collectionSelect.value;
    collectionSelect.innerHTML = (libraryMeta.collections || [])
      .map(item => `<option value="${escHtml(item.name)}">${escHtml(item.name)} (${item.count})</option>`)
      .join('');
    if ([...collectionSelect.options].some(option => option.value === value)) collectionSelect.value = value;
  }
  if (fileSelect) {
    const value = fileSelect.value;
    fileSelect.innerHTML = queryScopeFiles
      .map(item => `<option value="${item.id}">${escHtml(item.file_name)} · ${escHtml(item.collection || 'Inbox')}</option>`)
      .join('');
    if ([...fileSelect.options].some(option => option.value === value)) fileSelect.value = value;
  }
  updateQueryScopeControls();
}

function updateQueryScopeControls() {
  const mode = document.getElementById('query-scope-mode')?.value || 'all';
  const collectionSelect = document.getElementById('query-scope-collection');
  const fileSelect = document.getElementById('query-scope-file');
  const status = document.getElementById('query-scope-status');
  collectionSelect?.classList.toggle('hidden', mode !== 'collection');
  fileSelect?.classList.toggle('hidden', mode !== 'file');
  const label = queryScopeLabel();
  if (status) status.textContent = label;
  const contextScope = document.getElementById('chat-context-scope');
  if (contextScope) contextScope.textContent = label;
}

function buildQueryScopePayload() {
  const mode = document.getElementById('query-scope-mode')?.value || 'all';
  const payload = { scope_mode: mode, retrieval_mode: mode === 'full_text' ? 'full_text' : 'hybrid' };
  if (mode === 'collection') {
    payload.collection = document.getElementById('query-scope-collection')?.value || '';
  }
  if (mode === 'file') {
    const rawId = document.getElementById('query-scope-file')?.value || '';
    payload.file_id = rawId ? parseInt(rawId) : null;
  }
  return payload;
}

function validateQueryScopePayload(payload) {
  if (payload.scope_mode === 'collection' && !payload.collection) return '请选择集合';
  if (payload.scope_mode === 'file' && !payload.file_id) return '请选择文件';
  return '';
}

function queryScopeLabel() {
  const mode = document.getElementById('query-scope-mode')?.value || 'all';
  if (mode === 'collection') {
    return document.getElementById('query-scope-collection')?.value || '选择集合';
  }
  if (mode === 'file') {
    const select = document.getElementById('query-scope-file');
    return select?.selectedOptions?.[0]?.textContent || '选择文件';
  }
  if (mode === 'full_text') return '仅全文匹配';
  return `${queryScopeFiles.length} 个可提问文件`;
}

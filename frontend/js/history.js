// ── History page ──
let historyItems = [];

async function refreshHistory() {
  const listEl = document.getElementById('history-list');
  try {
    const items = await fetch(`${API}/api/history`).then(r => r.json());
    historyItems = items;
    if (!items.length) {
      listEl.innerHTML = '<div class="text-center text-on-surface-variant/60 text-sm py-16">暂无历史记录</div>';
      return;
    }
    listEl.innerHTML = items.map((item, idx) => `
      <div class="bg-surface-container-lowest rounded-xl p-4 cursor-pointer hover:bg-surface-container transition-all shadow-sm"
           role="button" tabindex="0" aria-label="打开历史记录：${escHtml(item.question || '未命名问题')}"
           onclick="replayHistory(${idx})"
           onkeydown="handleHistoryKey(event, ${idx})">
        <div class="text-sm font-semibold text-on-surface mb-2">${escHtml(item.question)}</div>
        <div class="text-xs text-on-surface-variant leading-relaxed mb-3 line-clamp-2">${escHtml(item.answer?.slice(0,150))}…</div>
        <div class="flex gap-4 text-[11px] text-on-surface-variant/60 font-medium">
          <span>${(item.created_at || '').slice(0,16)}</span>
          ${item.file_filter?.length ? `<span class="flex items-center gap-1"><span class="material-symbols-outlined" style="font-size:12px">folder_open</span>${escHtml(item.file_filter.join(', '))}</span>` : ''}
          ${item.citations?.length ? `<span class="flex items-center gap-1"><span class="material-symbols-outlined" style="font-size:12px">link</span>${item.citations.length} 引用</span>` : ''}
        </div>
      </div>`).join('');
  } catch (e) {
    listEl.innerHTML = `<div class="text-center text-error text-sm py-8">加载失败: ${e.message}</div>`;
  }
}

function replayHistory(idx) {
  const item = historyItems[idx];
  if (!item) return;
  currentConversationId = item.conversation_id || null;

  switchView('chat');

  // Replace chat messages with this historical Q&A
  const msgs = document.getElementById('messages');
  const inner = msgs.querySelector('.max-w-2xl');
  inner.innerHTML = '';

  // History context banner
  const banner = document.createElement('div');
  banner.className = 'flex items-center gap-2 text-[11px] font-bold uppercase tracking-widest text-on-surface-variant/50 pb-2';
  banner.innerHTML = `<span class="material-symbols-outlined" style="font-size:14px">history</span>历史记录 · ${(item.created_at || '').slice(0,16)}`;
  inner.appendChild(banner);

  // User message bubble
  const userDiv = document.createElement('div');
  userDiv.className = 'flex flex-col items-end gap-2';
  userDiv.innerHTML = `
    <div class="max-w-[85%] bg-primary text-on-primary px-5 py-3 rounded-2xl rounded-tr-sm shadow-sm">
      <p class="text-sm leading-relaxed">${escHtml(item.question)}</p>
    </div>`;
  inner.appendChild(userDiv);

  // AI answer bubble with citations
  const citationsHtml = citationMarkup(item.citations || []);

  const aiDiv = document.createElement('div');
  aiDiv.className = 'flex flex-col gap-4';
  aiDiv.innerHTML = `
    <div class="flex items-center gap-2">
      <div class="w-6 h-6 rounded-lg bg-surface-container-highest flex items-center justify-center">
        <span class="material-symbols-outlined text-primary icon-fill" style="font-size:13px">auto_awesome</span>
      </div>
      <span class="text-xs font-bold uppercase tracking-widest text-on-surface-variant">DocFlow</span>
    </div>
    <div class="prose text-sm text-on-surface max-w-none">${renderMarkdown(item.answer || '')}</div>
    <div class="flex flex-wrap gap-2">${citationsHtml}</div>
    <div class="flex gap-4 mt-1">
      <button class="answer-copy text-on-surface-variant/40 hover:text-primary transition-colors" data-copy-text="${escHtml(item.answer || '')}" onclick="copyTextFromButton(this)" title="复制答案" aria-label="复制答案">
        <span class="material-symbols-outlined" style="font-size:16px">content_copy</span>
      </button>
      <button class="answer-export text-on-surface-variant/40 hover:text-primary transition-colors" data-answer-text="${escHtml(item.answer || '')}" onclick="exportAnswerFromButton(this)" title="导出 Markdown" aria-label="导出 Markdown">
        <span class="material-symbols-outlined" style="font-size:16px">download</span>
      </button>
    </div>`;
  inner.appendChild(aiDiv);

  msgs.scrollTop = 0;

  // Pre-fill input so user can modify and resend
  const input = document.getElementById('input');
  input.value = item.question;
  autoResize(input);
  input.focus();
}

async function clearHistory() {
  const confirmed = await showConfirmDialog({
    title: '清空所有历史记录？',
    message: '这会删除历史页里的所有问答记录。对话列表不会被清空。',
    confirmText: '清空历史',
  });
  if (!confirmed) return;
  await fetch(`${API}/api/history`, { method: 'DELETE' });
  refreshHistory();
}

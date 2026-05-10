function appendUserMessage(question, scroll = true) {
  const msgs = document.getElementById('messages');
  const inner = msgs.querySelector('.max-w-2xl');
  if (inner.dataset.latestPreview === 'loaded' || inner.textContent.includes('你好，我是 DocFlow')) {
    inner.dataset.latestPreview = '';
    inner.innerHTML = '';
  }
  const div = document.createElement('div');
  div.className = 'flex flex-col items-end gap-2';
  div.innerHTML = `
    <div class="max-w-[85%] bg-primary text-on-primary px-5 py-3 rounded-2xl rounded-tr-sm shadow-sm">
      <p class="text-sm leading-relaxed">${escHtml(question)}</p>
    </div>
    <span class="text-[10px] font-bold text-on-surface-variant/40 tracking-wider">${new Date().toLocaleTimeString('zh-CN',{hour:'2-digit',minute:'2-digit'})}</span>`;
  inner.appendChild(div);
  if (scroll) msgs.scrollTop = msgs.scrollHeight;
}

function appendThinking() {
  const msgs = document.getElementById('messages');
  const inner = msgs.querySelector('.max-w-2xl');
  const div = document.createElement('div');
  div.id = 'thinking-indicator';
  div.className = 'flex flex-col gap-4';
  div.innerHTML = `
    <div class="flex items-center gap-2">
      <div class="w-6 h-6 rounded-lg bg-surface-container-highest flex items-center justify-center">
        <span class="material-symbols-outlined text-primary icon-fill" style="font-size:13px">auto_awesome</span>
      </div>
      <span class="text-xs font-bold uppercase tracking-widest text-on-surface-variant">DocFlow</span>
    </div>
    <div class="flex gap-1.5 items-center py-2">
      <span class="thinking-dot w-1.5 h-1.5 rounded-full bg-on-surface-variant/40"></span>
      <span class="thinking-dot w-1.5 h-1.5 rounded-full bg-on-surface-variant/40"></span>
      <span class="thinking-dot w-1.5 h-1.5 rounded-full bg-on-surface-variant/40"></span>
    </div>`;
  inner.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
}

function createAIMessageContainer() {
  const msgs = document.getElementById('messages');
  const inner = msgs.querySelector('.max-w-2xl');
  const div = document.createElement('div');
  div.className = 'flex flex-col gap-4';
  div.innerHTML = `
    <div class="flex items-center gap-2">
      <div class="w-6 h-6 rounded-lg bg-surface-container-highest flex items-center justify-center">
        <span class="material-symbols-outlined text-primary icon-fill" style="font-size:13px">auto_awesome</span>
      </div>
      <span class="text-xs font-bold uppercase tracking-widest text-on-surface-variant">DocFlow</span>
    </div>
    <div id="stream-prose" class="prose text-sm text-on-surface max-w-none"></div>
    <div id="stream-citations" class="flex flex-wrap gap-2"></div>
    <div id="stream-meta" class="text-[11px] text-on-surface-variant/50 font-medium"></div>
    <div class="flex gap-4 mt-1">
      <button onclick="copyTextFromButton(this)" data-copy-text="" class="answer-copy text-on-surface-variant/40 hover:text-primary transition-colors" title="复制答案" aria-label="复制答案">
        <span class="material-symbols-outlined" style="font-size:16px">content_copy</span>
      </button>
      <button onclick="saveAnswerFromButton(this)" data-answer-text="" class="answer-save text-on-surface-variant/40 hover:text-primary transition-colors" title="保存为笔记" aria-label="保存为笔记">
        <span class="material-symbols-outlined" style="font-size:16px">note_add</span>
      </button>
      <button onclick="exportAnswerFromButton(this)" data-answer-text="" class="answer-export text-on-surface-variant/40 hover:text-primary transition-colors" title="导出 Markdown" aria-label="导出 Markdown">
        <span class="material-symbols-outlined" style="font-size:16px">download</span>
      </button>
    </div>`;
  inner.appendChild(div);
  return div;
}

function citationScoreLabel(score) {
  const value = Number(score || 0);
  const pct = Math.round(value * 100);
  return pct > 0 ? `${pct}%` : '';
}

function citationMarkup(citations) {
  if (!citations?.length) return '';
  return citations.map(c => {
    const label = c.section ? escHtml(c.section) : `p.${c.page_num || '-'}`;
    const icon = (c.file_name || '').toLowerCase().endsWith('.md') ? 'article' : 'description';
    return `
    <div class="group relative flex items-center gap-2 px-3 py-2 bg-surface-container-low hover:bg-surface-container-high rounded-lg transition-all cursor-pointer"
         role="button" tabindex="0" aria-label="打开来源：${escHtml(c.file_name || '未知文件')}"
         onclick="openSourceByPath(decodeURIComponent('${encodeURIComponent(c.file_path || '')}'), ${c.page_num || 1})"
         onkeydown="handleSourceKey(event, decodeURIComponent('${encodeURIComponent(c.file_path || '')}'), ${c.page_num || 1})">
      <span class="material-symbols-outlined text-primary" style="font-size:15px">${icon}</span>
      <span class="text-xs font-medium text-on-surface-variant max-w-[160px] truncate">${escHtml(c.file_name)}</span>
      <span class="text-[10px] bg-primary/10 text-primary px-1.5 py-0.5 rounded font-bold whitespace-nowrap max-w-[150px] truncate">${label}</span>
      <div class="absolute bottom-full left-0 mb-2 w-64 p-3 bg-surface-container-lowest shadow-xl rounded-xl opacity-0 group-hover:opacity-100 pointer-events-none transition-all border border-outline-variant/10 z-50">
        ${c.section ? `<p class="text-[10px] font-bold text-primary mb-1">${escHtml(c.section)}</p>` : ''}
        <p class="text-[11px] leading-relaxed text-on-surface-variant italic">${escHtml(c.snippet?.slice(0,200))}</p>
      </div>
    </div>`;
  }).join('');
}

function renderCitations(citations) {
  const el = document.getElementById('stream-citations');
  if (!el || !citations.length) return;
  el.innerHTML = citationMarkup(citations);
  lastCitations = citations;
  renderChatContextSources(citations);
}

function appendAssistantMessage(answer, citations = [], meta = {}) {
  const msgs = document.getElementById('messages');
  const inner = msgs.querySelector('.max-w-2xl');
  const div = document.createElement('div');
  const copyText = escHtml(answer || '');
  const metaText = meta.elapsedMs ? `耗时 ${(meta.elapsedMs / 1000).toFixed(1)} 秒` : ((meta.createdAt || '').slice(0,16));
  div.className = 'flex flex-col gap-4';
  div.innerHTML = `
    <div class="flex items-center gap-2">
      <div class="w-6 h-6 rounded-lg bg-surface-container-highest flex items-center justify-center">
        <span class="material-symbols-outlined text-primary icon-fill" style="font-size:13px">auto_awesome</span>
      </div>
      <span class="text-xs font-bold uppercase tracking-widest text-on-surface-variant">DocFlow</span>
    </div>
    <div class="prose text-sm text-on-surface max-w-none">${renderMarkdown(answer || '')}</div>
    <div class="flex flex-wrap gap-2">${citationMarkup(citations)}</div>
    ${metaText ? `<div class="text-[11px] text-on-surface-variant/50 font-medium">${escHtml(metaText)}</div>` : ''}
    <div class="flex gap-4 mt-1">
      <button onclick="copyTextFromButton(this)" data-copy-text="${copyText}" class="answer-copy text-on-surface-variant/40 hover:text-primary transition-colors" title="复制答案" aria-label="复制答案">
        <span class="material-symbols-outlined" style="font-size:16px">content_copy</span>
      </button>
      <button onclick="saveAnswerFromButton(this)" data-answer-text="${copyText}" class="answer-save text-on-surface-variant/40 hover:text-primary transition-colors" title="保存为笔记" aria-label="保存为笔记">
        <span class="material-symbols-outlined" style="font-size:16px">note_add</span>
      </button>
      <button onclick="exportAnswerFromButton(this)" data-answer-text="${copyText}" class="answer-export text-on-surface-variant/40 hover:text-primary transition-colors" title="导出 Markdown" aria-label="导出 Markdown">
        <span class="material-symbols-outlined" style="font-size:16px">download</span>
      </button>
    </div>`;
  inner.appendChild(div);
  lastCitations = citations || [];
  renderChatContextSources(lastCitations);
  msgs.scrollTop = msgs.scrollHeight;
}

function renderChatContextSources(citations = lastCitations) {
  const list = document.getElementById('chat-context-sources');
  const count = document.getElementById('chat-context-source-count');
  const metric = document.getElementById('chat-context-source-metric');
  if (!list || !count) return;
  if (!citations?.length) {
    count.textContent = '等待提问后显示';
    if (metric) metric.textContent = '待生成';
    list.innerHTML = '<div class="rounded-lg bg-surface-container-low px-3 py-3">回答生成后，这里会显示本次引用的文件和片段。</div>';
    renderChatSourcePreview(null);
    return;
  }
  count.textContent = `${citations.length} 个来源`;
  if (metric) metric.textContent = `${citations.length} 个来源`;
  list.innerHTML = citations.slice(0, 6).map(c => `
    <button class="text-left rounded-lg bg-surface-container-low px-3 py-3 hover:bg-surface-container transition-colors"
      onclick="openSourceByPath(decodeURIComponent('${encodeURIComponent(c.file_path || '')}'), ${c.page_num || 1})"
      title="打开来源" aria-label="打开来源：${escHtml(c.file_name || '未知文件')}">
      <div class="flex items-center justify-between gap-2">
        <span class="font-semibold text-on-surface line-clamp-1">${escHtml(c.file_name || '未知文件')}</span>
        <span class="text-[11px] font-bold text-primary">${citationScoreLabel(c.score)}</span>
      </div>
      <div class="mt-1 text-[11px] text-on-surface-variant/60 line-clamp-2">${escHtml(c.section || c.snippet || '')}</div>
    </button>
  `).join('');
  renderChatSourcePreview(citations[0]);
  renderLocalIcons(list);
}

function renderChatSourcePreview(citation) {
  const panel = document.getElementById('chat-context-preview');
  const btn = document.getElementById('chat-source-open-btn');
  if (!panel || !btn) return;
  chatPreviewCitation = citation || null;
  if (!citation) {
    btn.classList.add('hidden');
    panel.innerHTML = '选择回答中的引用后，这里会显示可核查片段。';
    return;
  }
  btn.classList.remove('hidden');
  panel.innerHTML = `
    <div class="flex items-start gap-2">
      <span class="material-symbols-outlined text-primary" style="font-size:16px">${(citation.file_name || '').toLowerCase().endsWith('.md') ? 'article' : 'description'}</span>
      <div class="min-w-0">
        <div class="font-semibold text-on-surface line-clamp-1">${escHtml(citation.file_name || '未知文件')}</div>
        <div class="mt-0.5 text-[11px] text-on-surface-variant/60">${citation.page_num ? `第 ${citation.page_num} 页` : escHtml(citation.section || '来源片段')}</div>
      </div>
    </div>
    <p class="mt-3 line-clamp-5">${escHtml(citation.snippet || citation.section || '暂无片段预览')}</p>
    ${citationScoreLabel(citation.score) ? `<div class="mt-3 inline-flex items-center rounded-lg bg-primary-container px-2 py-1 text-[11px] font-bold text-primary">相关度 ${citationScoreLabel(citation.score)}</div>` : ''}`;
  renderLocalIcons(panel);
}

function openSourceFromChatPreview() {
  if (!chatPreviewCitation) return;
  openSourceByPath(chatPreviewCitation.file_path || '', chatPreviewCitation.page_num || 1);
}

function copyTextFromButton(btn) {
  const text = btn.dataset.copyText || '';
  if (text) {
    navigator.clipboard.writeText(text).then(() => {
      const icon = btn.querySelector('.material-symbols-outlined');
      setIcon(icon, 'check');
      setTimeout(() => { setIcon(icon, 'content_copy'); }, 1500);
    });
  }
}

function exportAnswerFromButton(btn) {
  const text = btn.dataset.answerText || '';
  if (!text) return;
  const blob = new Blob([text], { type: 'text/markdown;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `docflow-answer-${new Date().toISOString().slice(0,10)}.md`;
  a.click();
  URL.revokeObjectURL(url);
}

async function saveAnswerFromButton(btn) {
  const answer = btn.dataset.answerText || '';
  if (!answer.trim()) return;
  btn.disabled = true;
  const icon = btn.querySelector('.material-symbols-outlined');
  const previousIcon = getIconToken(icon);
  setIcon(icon, 'sync');
  icon.classList.add('animate-spin');
  try {
    const r = await fetch(`${API}/api/notes/from-answer`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        title: `Saved Answer ${new Date().toISOString().slice(0,10)}`,
        answer,
        collection: 'Saved Answers',
        user_tags: ['answer'],
      }),
    });
    if (!r.ok) throw new Error(await r.text());
    icon.classList.remove('animate-spin');
    setIcon(icon, 'check');
    setTimeout(() => { setIcon(icon, previousIcon); }, 1500);
    if (!document.getElementById('view-library').classList.contains('hidden')) refreshFiles();
  } catch (e) {
    icon.classList.remove('animate-spin');
    setIcon(icon, 'error');
    alert(`保存笔记失败: ${e.message}`);
    setTimeout(() => { setIcon(icon, previousIcon); }, 1500);
  } finally {
    btn.disabled = false;
  }
}

async function sendMessage() {
  const input = document.getElementById('input');
  const question = input.value.trim();
  if (!question) return;
  const scopePayload = buildQueryScopePayload();
  const scopeError = validateQueryScopePayload(scopePayload);
  if (scopeError) {
    alert(scopeError);
    return;
  }

  input.value = '';
  input.style.height = 'auto';
  document.getElementById('send-btn').disabled = true;

  appendUserMessage(question);
  appendThinking();
  const startedAt = performance.now();

  try {
    const body = { question, conversation_id: currentConversationId, ...scopePayload };
    const r = await fetch(`${API}/api/query/stream`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body),
    });
    if (!r.ok || !r.body) {
      throw new Error(await responseUserMessage(r, '本次查询失败，请稍后再试。'));
    }

    document.getElementById('thinking-indicator')?.remove();
    const msgContainer = createAIMessageContainer();
    const msgs = document.getElementById('messages');
    const prose = document.getElementById('stream-prose');
    const meta = document.getElementById('stream-meta');

    let answerText = '';
    const reader = r.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let streamCompleted = false;
    let streamErrored = false;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const parts = buffer.split('\n\n');
      buffer = parts.pop();

      for (const part of parts) {
        const lines = part.split('\n');
        let eventType = '', eventData = '';
        for (const line of lines) {
          if (line.startsWith('event: ')) eventType = line.slice(7);
          else if (line.startsWith('data: ')) eventData = line.slice(6);
        }
        if (!eventType) continue;

        if (eventType === 'conversation') {
          const payload = JSON.parse(eventData);
          currentConversationId = payload.conversation_id;
        } else if (eventType === 'citations') {
          renderCitations(JSON.parse(eventData));
        } else if (eventType === 'token') {
          const token = JSON.parse(eventData);
          answerText += token;
          prose.classList.add('streaming-cursor');
          prose.innerHTML = renderMarkdown(answerText);
          msgs.scrollTop = msgs.scrollHeight;
        } else if (eventType === 'done') {
          streamCompleted = true;
          prose.classList.remove('streaming-cursor');
          const elapsedMs = performance.now() - startedAt;
          if (meta) meta.textContent = `耗时 ${(elapsedMs / 1000).toFixed(1)} 秒`;
          msgContainer.querySelector('.answer-copy').dataset.copyText = answerText;
          msgContainer.querySelector('.answer-save').dataset.answerText = answerText;
          msgContainer.querySelector('.answer-export').dataset.answerText = answerText;
          loadConversations();
        } else if (eventType === 'error') {
          streamErrored = true;
          prose.classList.remove('streaming-cursor');
          let rawError = eventData;
          try { rawError = JSON.parse(eventData); } catch {}
          prose.innerHTML = `<span class="text-error">${escHtml(userFacingErrorMessage(rawError, '本次回答失败，请稍后再试。'))}</span>`;
          if (meta) meta.textContent = '回答失败';
        }
      }
    }

    prose.classList.remove('streaming-cursor');
    if (!streamCompleted && !streamErrored) {
      const interrupted = '连接已中断，以上内容可能不完整。';
      answerText = answerText ? `${answerText}\n\n${interrupted}` : interrupted;
      prose.innerHTML = renderMarkdown(answerText);
      msgContainer.querySelector('.answer-copy').dataset.copyText = answerText;
      msgContainer.querySelector('.answer-save').dataset.answerText = answerText;
      msgContainer.querySelector('.answer-export').dataset.answerText = answerText;
      if (meta) meta.textContent = '连接中断';
    }
    prose.removeAttribute('id');
    document.getElementById('stream-citations')?.removeAttribute('id');
  } catch (e) {
    document.getElementById('thinking-indicator')?.remove();
    const msgs = document.getElementById('messages');
    const inner = msgs.querySelector('.max-w-2xl');
    const div = document.createElement('div');
    div.className = 'rounded-xl bg-surface-container-low border border-error/20 px-4 py-3 text-sm text-error';
    div.textContent = userFacingErrorMessage(e.message, '本次查询失败，请稍后再试。');
    inner.appendChild(div);
  } finally {
    document.getElementById('send-btn').disabled = false;
    input.focus();
  }
}

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

function feedbackControlsMarkup(historyId) {
  const id = Number(historyId || 0);
  if (!id) return '';
  return `
    <span class="answer-feedback inline-flex items-center gap-1" data-history-id="${id}">
      <button onclick="submitAnswerFeedback(this, ${id}, 'useful')" class="answer-action" title="这次回答有帮助" aria-label="这次回答有帮助">
        <span class="material-symbols-outlined" style="font-size:15px">thumb_up</span>有用
      </button>
      <button onclick="submitAnswerFeedback(this, ${id}, 'not_useful')" class="answer-action" title="这次回答需要改进" aria-label="这次回答需要改进">
        <span class="material-symbols-outlined" style="font-size:15px">thumb_down</span>需要改进
      </button>
    </span>`;
}

async function submitAnswerFeedback(btn, historyId, rating) {
  const group = btn.closest('.answer-feedback');
  const buttons = [...(group?.querySelectorAll('button') || [])];
  buttons.forEach(item => { item.disabled = true; });
  const icon = btn.querySelector('.material-symbols-outlined');
  const previousIcon = getIconToken(icon);
  setIcon(icon, 'sync');
  icon.classList.add('animate-spin');
  try {
    const r = await fetch(`${API}/api/answers/feedback`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ history_id: historyId, rating }),
    });
    if (!r.ok) throw new Error(await responseUserMessage(r, '反馈保存失败'));
    icon.classList.remove('animate-spin');
    setIcon(icon, 'check');
    if (group) {
      group.dataset.rating = rating;
      buttons.forEach(item => item.classList.remove('bg-primary/10', 'text-primary'));
      btn.classList.add('bg-primary/10', 'text-primary');
    }
  } catch (e) {
    icon.classList.remove('animate-spin');
    setIcon(icon, 'error');
    alert(userFacingErrorMessage(e.message, '反馈保存失败，请稍后再试。'));
    setTimeout(() => { setIcon(icon, previousIcon); }, 1500);
  } finally {
    buttons.forEach(item => { item.disabled = false; });
  }
}

function createAIMessageContainer(question = '') {
  const msgs = document.getElementById('messages');
  const inner = msgs.querySelector('.max-w-2xl');
  const div = document.createElement('div');
  div.className = 'flex flex-col gap-4';
  const questionText = escHtml(question || '');
  div.innerHTML = `
    <div class="flex items-center gap-2">
      <div class="w-6 h-6 rounded-lg bg-surface-container-highest flex items-center justify-center">
        <span class="material-symbols-outlined text-primary icon-fill" style="font-size:13px">auto_awesome</span>
      </div>
      <span class="text-xs font-bold uppercase tracking-widest text-on-surface-variant">DocFlow</span>
    </div>
    <div id="stream-prose" class="prose text-sm text-on-surface max-w-none"></div>
    <div id="stream-citations" class="flex flex-wrap gap-2"></div>
    <div id="stream-evidence"></div>
    <div id="stream-related-notes"></div>
    <div id="stream-meta" class="text-[11px] text-on-surface-variant/50 font-medium"></div>
    <div class="flex gap-4 mt-1">
      <button onclick="copyTextFromButton(this)" data-copy-text="" class="answer-copy text-on-surface-variant/40 hover:text-primary transition-colors" title="复制答案" aria-label="复制答案">
        <span class="material-symbols-outlined" style="font-size:16px">content_copy</span>
      </button>
      <button onclick="saveAnswerFromButton(this)" data-answer-text="" data-question-text="${questionText}" data-citations-json="" class="answer-save text-on-surface-variant/40 hover:text-primary transition-colors" title="保存为笔记" aria-label="保存为笔记">
        <span class="material-symbols-outlined" style="font-size:16px">note_add</span>
      </button>
      <button onclick="exportAnswerFromButton(this)" data-answer-text="" class="answer-export text-on-surface-variant/40 hover:text-primary transition-colors" title="导出 Markdown" aria-label="导出 Markdown">
        <span class="material-symbols-outlined" style="font-size:16px">download</span>
      </button>
      <span class="answer-feedback"></span>
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
    const chunkLabel = c.chunk_id ? `片段 ${escHtml(c.chunk_id)}` : '';
    return `
    <div class="group relative flex items-center gap-2 px-3 py-2 bg-surface-container-low hover:bg-surface-container-high rounded-lg transition-all cursor-pointer"
         role="button" tabindex="0" aria-label="打开来源：${escHtml(c.file_name || '未知文件')}"
         data-chunk-id="${escHtml(c.chunk_id || '')}"
         onclick="openSourceByPath(decodeURIComponent('${encodeURIComponent(c.file_path || '')}'), ${c.page_num || 1}, '${escHtml(c.chunk_id || '')}', ${Number(c.char_start || 0)}, ${Number(c.char_end || 0)})"
         onkeydown="handleSourceKey(event, decodeURIComponent('${encodeURIComponent(c.file_path || '')}'), ${c.page_num || 1}, '${escHtml(c.chunk_id || '')}', ${Number(c.char_start || 0)}, ${Number(c.char_end || 0)})">
      <span class="material-symbols-outlined text-primary" style="font-size:15px">${icon}</span>
      <span class="text-xs font-medium text-on-surface-variant max-w-[160px] truncate">${escHtml(c.file_name)}</span>
      <span class="text-[10px] bg-primary/10 text-primary px-1.5 py-0.5 rounded font-bold whitespace-nowrap max-w-[150px] truncate">${label}</span>
      ${citationEvidencePill(c)}
      <div class="absolute bottom-full left-0 mb-2 w-64 p-3 bg-surface-container-lowest shadow-xl rounded-xl opacity-0 group-hover:opacity-100 pointer-events-none transition-all border border-outline-variant/10 z-50">
        ${c.section ? `<p class="text-[10px] font-bold text-primary mb-1">${escHtml(c.section)}</p>` : ''}
        ${c.evidence_reason ? `<p class="text-[10px] text-primary mb-1">${escHtml(citationEvidenceReason(c))}</p>` : ''}
        ${chunkLabel ? `<p class="text-[10px] text-on-surface-variant/70 mb-1">${chunkLabel}</p>` : ''}
        <p class="text-[11px] leading-relaxed text-on-surface-variant italic">${escHtml(c.snippet?.slice(0,200))}</p>
      </div>
    </div>`;
  }).join('');
}

function relatedNotesMarkup(notes) {
  if (!notes?.length) return '';
  return notes.map(note => {
    const icon = (note.file_name || '').toLowerCase().endsWith('.md') ? 'article' : 'description';
    const label = note.section || (note.page_num ? `p.${note.page_num}` : '相关片段');
    return `
    <button class="text-left rounded-lg bg-surface-container-low px-3 py-2 hover:bg-surface-container transition-colors"
      onclick="openSourceByPath(decodeURIComponent('${encodeURIComponent(note.file_path || '')}'), ${note.page_num || 1})"
      title="打开相关笔记" aria-label="打开相关笔记：${escHtml(note.file_name || '未知文件')}">
      <div class="flex items-center justify-between gap-2">
        <span class="flex min-w-0 items-center gap-2">
          <span class="material-symbols-outlined text-primary" style="font-size:15px">${icon}</span>
          <span class="font-semibold text-on-surface line-clamp-1">${escHtml(note.file_name || '未知文件')}</span>
        </span>
        <span class="text-[10px] bg-primary/10 text-primary px-1.5 py-0.5 rounded font-bold whitespace-nowrap max-w-[120px] truncate">${escHtml(label)}</span>
      </div>
      <div class="mt-1 text-[11px] text-on-surface-variant/60 line-clamp-2">${escHtml(note.snippet || '')}</div>
    </button>`;
  }).join('');
}

function renderCitations(citations) {
  const el = document.getElementById('stream-citations');
  if (!el || !citations.length) return;
  el.innerHTML = citationMarkup(citations);
  lastCitations = citations;
  renderChatContextSources(citations);
}

function renderRelatedNotes(notes = lastRelatedNotes) {
  const list = document.getElementById('chat-related-notes');
  const count = document.getElementById('chat-related-count');
  if (!list || !count) return;
  lastRelatedNotes = notes || [];
  if (!lastRelatedNotes.length) {
    count.textContent = '暂无相关笔记';
    list.innerHTML = '<div class="rounded-lg bg-surface-container-low px-3 py-3">这次回答没有额外的相关笔记。</div>';
    return;
  }
  count.textContent = `${lastRelatedNotes.length} 条可探索`;
  list.innerHTML = relatedNotesMarkup(lastRelatedNotes.slice(0, 4));
  renderLocalIcons(list);
}

function appendAssistantMessage(answer, citations = [], meta = {}) {
  const msgs = document.getElementById('messages');
  const inner = msgs.querySelector('.max-w-2xl');
  const div = document.createElement('div');
  const copyText = escHtml(answer || '');
  const metaText = meta.elapsedMs ? `耗时 ${(meta.elapsedMs / 1000).toFixed(1)} 秒` : ((meta.createdAt || '').slice(0,16));
  const citationsJson = encodeURIComponent(JSON.stringify(citations || []));
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
    ${evidenceSummaryMarkup(meta.evidence)}
    ${meta.relatedNotes?.length ? `<div class="rounded-xl bg-surface-container-low px-4 py-3">
      <div class="mb-2 flex items-center gap-2 text-xs font-bold text-on-surface">
        <span class="material-symbols-outlined text-primary" style="font-size:15px">hub</span>相关笔记
      </div>
      <div class="grid gap-2">${relatedNotesMarkup(meta.relatedNotes)}</div>
    </div>` : ''}
    ${metaText ? `<div class="text-[11px] text-on-surface-variant/50 font-medium">${escHtml(metaText)}</div>` : ''}
    <div class="flex gap-4 mt-1">
      <button onclick="copyTextFromButton(this)" data-copy-text="${copyText}" class="answer-copy text-on-surface-variant/40 hover:text-primary transition-colors" title="复制答案" aria-label="复制答案">
        <span class="material-symbols-outlined" style="font-size:16px">content_copy</span>
      </button>
      <button onclick="saveAnswerFromButton(this)" data-answer-text="${copyText}" data-question-text="${escHtml(meta.question || '')}" data-citations-json="${escHtml(citationsJson)}" class="answer-save text-on-surface-variant/40 hover:text-primary transition-colors" title="保存为笔记" aria-label="保存为笔记">
        <span class="material-symbols-outlined" style="font-size:16px">note_add</span>
      </button>
      <button onclick="exportAnswerFromButton(this)" data-answer-text="${copyText}" class="answer-export text-on-surface-variant/40 hover:text-primary transition-colors" title="导出 Markdown" aria-label="导出 Markdown">
        <span class="material-symbols-outlined" style="font-size:16px">download</span>
      </button>
      ${feedbackControlsMarkup(meta.historyId)}
    </div>`;
  inner.appendChild(div);
  lastCitations = citations || [];
  lastRelatedNotes = meta.relatedNotes || [];
  renderChatContextSources(lastCitations);
  renderRelatedNotes(lastRelatedNotes);
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
      onclick="openSourceByPath(decodeURIComponent('${encodeURIComponent(c.file_path || '')}'), ${c.page_num || 1}, '${escHtml(c.chunk_id || '')}', ${Number(c.char_start || 0)}, ${Number(c.char_end || 0)})"
      title="打开来源" aria-label="打开来源：${escHtml(c.file_name || '未知文件')}">
      <div class="flex items-center justify-between gap-2">
        <span class="font-semibold text-on-surface line-clamp-1">${escHtml(c.file_name || '未知文件')}</span>
        <span class="flex items-center gap-1">${citationEvidencePill(c)}<span class="text-[11px] font-bold text-primary">${citationScoreLabel(c.score)}</span></span>
      </div>
      <div class="mt-1 text-[11px] text-on-surface-variant/60 line-clamp-2">${escHtml(c.section || c.snippet || '')}</div>
      ${c.chunk_id ? `<div class="mt-1 text-[10px] text-on-surface-variant/50">片段 ${escHtml(c.chunk_id)}</div>` : ''}
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
        ${citation.chunk_id ? `<div class="mt-0.5 text-[10px] text-on-surface-variant/50">片段 ${escHtml(citation.chunk_id)} · ${citation.char_start || 0}-${citation.char_end || 0}</div>` : ''}
      </div>
    </div>
    <p class="mt-3 line-clamp-5">${escHtml(citation.snippet || citation.section || '暂无片段预览')}</p>
    ${citation.evidence_reason ? `<div class="mt-3 rounded-lg bg-surface-container px-3 py-2 text-[11px] font-medium text-on-surface-variant">${escHtml(citationEvidenceReason(citation))}</div>` : ''}
    ${citationScoreLabel(citation.score) ? `<div class="mt-3 inline-flex items-center rounded-lg bg-primary-container px-2 py-1 text-[11px] font-bold text-primary">相关度 ${citationScoreLabel(citation.score)}</div>` : ''}`;
  renderLocalIcons(panel);
}

function openSourceFromChatPreview() {
  if (!chatPreviewCitation) return;
  openSourceByPath(
    chatPreviewCitation.file_path || '',
    chatPreviewCitation.page_num || 1,
    chatPreviewCitation.chunk_id || '',
    Number(chatPreviewCitation.char_start || 0),
    Number(chatPreviewCitation.char_end || 0),
  );
}

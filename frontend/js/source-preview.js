// ── Source Preview ──
async function openSourceByPath(filePath, pageNum = 1, chunkId = '', charStart = 0, charEnd = 0) {
  try {
    const files = await fetch(`${API}/api/files`).then(r => r.json());
    const file = files.find(item => item.file_path === filePath);
    if (file) await openSourcePreview(file.id, pageNum, { chunkId, charStart, charEnd });
  } catch {}
}

async function loadDefaultSourcePreview() {
  if (sourcePreviewState.loading || sourcePreviewState.file || sourcePreviewState.error) {
    renderSourcePreview();
    return;
  }
  try {
    const files = await fetch(`${API}/api/files?status=done`).then(r => r.json());
    const sourceScore = file => {
      const chunks = Number(file?.chunk_count || 0);
      const pages = Number(file?.total_pages || 0);
      const tooFragmentedPenalty = chunks > 180 ? 200 : 0;
      return Math.min(chunks, 120) + pages * 4 - tooFragmentedPenalty;
    };
    const file = Array.isArray(files)
      ? [...files].sort((a, b) => sourceScore(b) - sourceScore(a))[0]
      : null;
    if (file?.id) {
      await openSourcePreview(file.id);
      return;
    }
  } catch {}
  renderSourcePreview();
}

async function openSourcePreview(fileId, pageNum = 1, highlight = null) {
  sourcePreviewState = {
    file: null,
    chunks: [],
    selectedIndex: 0,
    loading: true,
    error: '',
    highlight,
  };
  switchView('source');
  renderSourcePreview();
  try {
    const data = await fetch(`${API}/api/file/${fileId}/chunks?max_text_chars=1800`).then(r => {
      if (!r.ok) throw new Error('来源读取失败');
      return r.json();
    });
    const chunks = data.chunks || [];
    const targetQdrantId = qdrantIdFromChunkId(highlight?.chunkId);
    const chunkIndex = targetQdrantId === null
      ? -1
      : chunks.findIndex(chunk => Number(chunk.qdrant_id) === targetQdrantId);
    const pageIndex = chunks.findIndex(
      chunk => Number(chunk.page_num || 0) === Number(pageNum || 0),
    );
    const selectedIndex = chunkIndex >= 0 ? chunkIndex : Math.max(0, pageIndex);
    sourcePreviewState = {
      file: data.file,
      chunks,
      selectedIndex: selectedIndex >= 0 ? selectedIndex : 0,
      loading: false,
      error: '',
      highlight,
    };
  } catch (e) {
    sourcePreviewState = {
      file: null,
      chunks: [],
      selectedIndex: 0,
      loading: false,
      error: e.message,
      highlight: null,
    };
  }
  renderSourcePreview();
}

function qdrantIdFromChunkId(chunkId) {
  const match = String(chunkId || '').match(/^q:(\d+)$/);
  return match ? Number(match[1]) : null;
}

function sourceConfidence(chunk, index) {
  const raw = Number(chunk?.score || chunk?.similarity || 0);
  if (raw > 0) return Math.min(0.99, raw);
  return null;
}

function confidenceLabel(value) {
  if (value === null || value === undefined) return '来源片段';
  if (value >= 0.88) return '高置信度';
  if (value >= 0.65) return '中等置信度';
  return '低置信度';
}

function sourceKeywords(text) {
  const tokens = String(text || '').match(/[A-Za-z0-9_\u4e00-\u9fa5]{2,14}/g) || [];
  const blocked = new Set(['the','and','for','with','this','that','from','一个','可以','以及','这些','我们','进行','内容','文件']);
  return [...new Set(tokens.filter(token => !blocked.has(token.toLowerCase())))].slice(0, 10);
}

function sourceChunkText(chunk) {
  const values = [chunk?.parent_text, chunk?.raw_text, chunk?.text_preview];
  return values.find(value => value && !String(value).includes('暂无文本预览')) || chunk?.text_preview || '';
}

function sourcePreviewListTitle(chunk, index) {
  const section = String(chunk?.section || '').trim();
  if (section) return section;
  const text = sourceChunkText(chunk).replace(/\s+/g, ' ').trim();
  if (text) return text.slice(0, 32);
  return `片段 ${index + 1}`;
}

function highlightRangeApplies(chunk, highlight) {
  if (!chunk || !highlight?.chunkId) return false;
  const targetQdrantId = qdrantIdFromChunkId(highlight.chunkId);
  return targetQdrantId !== null && Number(chunk.qdrant_id) === targetQdrantId;
}

function highlightedSourceText(text, highlight = null) {
  const value = String(text || '暂无文本预览');
  if (highlight && Number(highlight.charEnd) > Number(highlight.charStart)) {
    const start = Math.max(0, Math.min(value.length, Number(highlight.charStart)));
    const end = Math.max(start, Math.min(value.length, Number(highlight.charEnd)));
    const body = [
      escHtml(value.slice(0, start)),
      `<mark class="source-highlight" data-citation-hit="true">${escHtml(value.slice(start, end))}</mark>`,
      escHtml(value.slice(end)),
    ].join('');
    return body
      .split(/\n{2,}/)
      .map(paragraph => `<p>${paragraph.replace(/\n/g, '<br>')}</p>`)
      .join('');
  }
  const lines = value.split(/\n+/).map(line => line.trim()).filter(Boolean).slice(0, 14);
  return lines.map((line, idx) => {
    const body = escHtml(line);
    if (idx < 5) return `<p><span class="source-highlight">${body}</span></p>`;
    return `<p>${body}</p>`;
  }).join('');
}

function renderSourceTimeline(chunks, selectedIndex) {
  if (!chunks.length) return '';
  const selected = chunks[selectedIndex] || chunks[0];
  const pageNums = chunks.map(chunk => Number(chunk.page_num || 0)).filter(Boolean);
  const first = pageNums.length ? Math.min(...pageNums) : 1;
  const last = pageNums.length ? Math.max(...pageNums) : chunks.length;
  const selectedPage = Number(selected?.page_num || selectedIndex + 1);
  return `
    <div class="mt-5 rounded-lg bg-surface-container-lowest px-4 py-3 shadow-sm">
      <div class="flex items-center justify-between text-xs">
        <span class="font-semibold text-on-surface">来源时间线</span>
        <span class="text-on-surface-variant/60">第 ${selectedPage || selectedIndex + 1} 页</span>
      </div>
      <div class="mt-4 flex items-center gap-2">
        ${chunks.slice(0, 12).map((chunk, idx) => `<button onclick="selectSourcePreviewChunk(${idx})" class="${idx === selectedIndex ? 'source-timeline-dot' : 'w-2 h-2 rounded-full bg-on-surface-variant/25'}" title="片段 ${idx + 1}" aria-label="片段 ${idx + 1}"></button>`).join('')}
      </div>
      <div class="mt-3 flex justify-between text-[11px] text-on-surface-variant/60">
        <span>${first}</span>
        <span>${selectedPage || selectedIndex + 1}</span>
        <span>${last}</span>
      </div>
    </div>`;
}

function renderSourcePreview() {
  const list = document.getElementById('source-result-list');
  const count = document.getElementById('source-preview-count');
  const title = document.getElementById('source-document-title');
  const meta = document.getElementById('source-document-meta');
  const viewer = document.getElementById('source-document-viewer');
  const detail = document.getElementById('source-detail-panel');
  const openOriginal = document.getElementById('source-open-original-btn');
  if (!list || !count || !title || !meta || !viewer || !detail || !openOriginal) return;

  if (sourcePreviewState.loading) {
    count.textContent = '读取中';
    title.textContent = '正在载入来源';
    meta.textContent = '读取本地片段';
    openOriginal.classList.add('hidden');
    list.innerHTML = '<div class="rounded-lg bg-surface-container-low px-3 py-3"><span class="spinner"></span><span class="ml-2">正在读取片段…</span></div>';
    viewer.innerHTML = '<div class="mx-auto max-w-2xl rounded-xl bg-surface-container-low px-5 py-8 text-center text-on-surface-variant">正在载入来源内容…</div>';
    detail.textContent = '片段载入完成后显示详情。';
    return;
  }

  if (sourcePreviewState.error) {
    count.textContent = '读取失败';
    title.textContent = '来源读取失败';
    meta.textContent = '';
    openOriginal.classList.add('hidden');
    list.innerHTML = `<div class="rounded-lg bg-error/10 px-3 py-3 text-error">${escHtml(sourcePreviewState.error)}</div>`;
    viewer.innerHTML = '<div class="mx-auto max-w-2xl rounded-xl bg-error/10 px-5 py-8 text-center text-error">无法读取这个来源。</div>';
    detail.textContent = '可以回到资料库重新选择文件。';
    return;
  }

  const file = sourcePreviewState.file;
  const chunks = sourcePreviewState.chunks || [];
  if (!file) {
    count.textContent = '0 个片段';
    title.textContent = '选择一个来源';
    meta.textContent = '等待载入片段';
    openOriginal.classList.add('hidden');
    list.innerHTML = '<div class="rounded-lg bg-surface-container-low px-3 py-3">从对话引用或资料库文件进入来源预览。</div>';
    viewer.innerHTML = '<div class="mx-auto max-w-2xl rounded-xl bg-surface-container-low px-5 py-8 text-center text-on-surface-variant">从对话引用或资料库文件进入来源预览。</div>';
    detail.textContent = '选中左侧片段后，这里会显示页码、章节、字数和可执行操作。';
    return;
  }

  const selected = chunks[sourcePreviewState.selectedIndex] || null;
  count.textContent = `${chunks.length} 个片段`;
  title.textContent = file.file_name || '未命名来源';
  meta.textContent = `${fileLocationLabel(file)} · ${fileTypeLabel(file.file_name, file.is_scanned)}`;
  openOriginal.classList.remove('hidden');

  if (!chunks.length) {
    list.innerHTML = '<div class="rounded-lg bg-surface-container-low px-3 py-3">这个文件还没有可预览片段。</div>';
    viewer.innerHTML = '<div class="mx-auto max-w-2xl rounded-xl bg-surface-container-low px-5 py-8 text-center text-on-surface-variant">这个文件还没有完成入库，暂时没有片段可预览。</div>';
    detail.textContent = '文件完成入库后会显示可核查片段。';
    return;
  }

  list.innerHTML = chunks.map((chunk, index) => {
    const active = index === sourcePreviewState.selectedIndex;
    const confidence = sourceConfidence(chunk, index);
    const itemTitle = sourcePreviewListTitle(chunk, index);
    const itemMeta = `${escHtml(file.file_name || '来源')} · ${chunk.page_num ? `第 ${chunk.page_num} 页` : `片段 ${index + 1}`}`;
    return `<button onclick="selectSourcePreviewChunk(${index})" class="w-full text-left rounded-lg px-3 py-3 transition-all active:scale-95 ${active ? 'bg-primary-container text-on-primary-container shadow-sm' : 'bg-surface-container-low hover:bg-surface-container'}">
      <div class="flex items-start gap-3">
        <span class="mt-0.5 w-5 h-5 rounded-md bg-surface-container-lowest flex items-center justify-center text-[11px] font-bold">${index + 1}</span>
        <div class="min-w-0 flex-1">
          <div class="flex items-center justify-between gap-2">
            <span class="font-semibold line-clamp-1">${escHtml(itemTitle)}</span>
            <span class="material-symbols-outlined" style="font-size:14px">arrow_forward</span>
          </div>
          <div class="mt-1 text-[11px] text-on-surface-variant/70 line-clamp-1">${itemMeta}</div>
          <div class="mt-2 inline-flex items-center gap-1 rounded-lg bg-surface-container-lowest px-2 py-1 text-[11px] font-bold text-primary">${confidenceLabel(confidence)}${confidence ? ` ${confidence.toFixed(2)}` : ''}</div>
        </div>
      </div>
    </button>`;
  }).join('');

  const activeHighlight = highlightRangeApplies(selected, sourcePreviewState.highlight)
    ? sourcePreviewState.highlight
    : null;
  const text = sourceChunkText(selected) || '暂无文本预览';
  const confidence = sourceConfidence(selected, sourcePreviewState.selectedIndex);
  const keywords = sourceKeywords(text);
  const selectedPage = selected?.page_num || sourcePreviewState.selectedIndex + 1;
  viewer.innerHTML = `
    <div class="mx-auto max-w-3xl">
      <div class="mb-4 flex items-center justify-between gap-3 rounded-lg bg-surface-container-lowest px-3 py-2 shadow-sm">
        <div class="flex items-center gap-2 text-xs">
          <button onclick="selectSourcePreviewChunk(${Math.max(0, sourcePreviewState.selectedIndex - 1)})" class="icon-button !w-8 !h-8" title="上一个片段" aria-label="上一个片段">
            <span class="material-symbols-outlined" style="font-size:14px">arrow_back</span>
          </button>
          <span class="rounded-lg bg-surface-container-low px-3 py-1 font-bold text-on-surface">第 ${selectedPage} 页</span>
          <button onclick="selectSourcePreviewChunk(${Math.min(chunks.length - 1, sourcePreviewState.selectedIndex + 1)})" class="icon-button !w-8 !h-8" title="下一个片段" aria-label="下一个片段">
            <span class="material-symbols-outlined" style="font-size:14px">arrow_forward</span>
          </button>
        </div>
        <div class="flex items-center gap-2 text-xs">
          <span class="rounded-lg bg-surface-container-low px-3 py-1 font-bold">100%</span>
          <button onclick="savePreviewChunkAsNote()" class="toolbar-btn" title="保存片段为笔记" aria-label="保存片段为笔记">
            <span class="material-symbols-outlined" style="font-size:14px">note_add</span>保存
          </button>
        </div>
      </div>
      <article class="rounded-xl bg-surface-container-lowest px-8 py-7 shadow-sm">
        <div class="text-[11px] font-bold uppercase tracking-widest text-on-surface-variant/60">${escHtml(selected?.section || selected?.chunk_type || '引用片段')}</div>
        <div class="mt-4 prose text-sm text-on-surface max-w-none">${highlightedSourceText(text, activeHighlight)}</div>
      </article>
      ${renderSourceTimeline(chunks, sourcePreviewState.selectedIndex)}
    </div>`;
  detail.innerHTML = `
    <div class="grid grid-cols-2 gap-2">
      <div class="rounded-lg bg-surface-container-low px-3 py-2">
        <div class="text-[10px] text-on-surface-variant/60">相似度</div>
        <div class="mt-1 font-bold text-on-surface">${confidence ? confidence.toFixed(2) : '-'}</div>
      </div>
      <div class="rounded-lg bg-surface-container-low px-3 py-2">
        <div class="text-[10px] text-on-surface-variant/60">证据强度</div>
        <div class="mt-1 font-bold text-primary">${confidence ? (confidence >= 0.75 ? '强' : '中') : '可核查'}</div>
      </div>
      <div class="rounded-lg bg-surface-container-low px-3 py-2">
        <div class="text-[10px] text-on-surface-variant/60">位置权重</div>
        <div class="mt-1 font-bold text-on-surface">${selected?.section ? '高' : '中'}</div>
      </div>
      <div class="rounded-lg bg-surface-container-low px-3 py-2">
        <div class="text-[10px] text-on-surface-variant/60">字数</div>
        <div class="mt-1 font-bold text-on-surface">${selected?.char_count || selected?.text_length || text.length}</div>
      </div>
    </div>
    <div class="mt-3 rounded-lg bg-surface-container-low px-3 py-2">
      <div class="font-semibold text-on-surface">引用关系</div>
      <div class="mt-2 text-[11px] leading-relaxed text-on-surface-variant/70">${activeHighlight ? '已定位到回答引用的原文范围。' : '这个片段来自当前文件的可核查内容，可用于回答中对应事实和段落。'}</div>
      <div class="mt-3 grid grid-cols-1 gap-2">
        <div class="rounded-lg bg-surface-container-lowest px-3 py-2">
          <div class="text-[10px] text-on-surface-variant/60">贡献内容</div>
          <div class="mt-1 text-on-surface line-clamp-2">${escHtml(text.slice(0, 120))}</div>
        </div>
        <div class="rounded-lg bg-surface-container-lowest px-3 py-2">
          <div class="text-[10px] text-on-surface-variant/60">在答案中的位置</div>
          <div class="mt-1 text-on-surface">第 ${sourcePreviewState.selectedIndex + 1} 段 · ${escHtml(selected?.section || '正文')}</div>
        </div>
      </div>
    </div>
    <div class="mt-3 rounded-lg bg-surface-container-low px-3 py-2">
      <div class="font-semibold text-on-surface">关键词命中</div>
      <div class="mt-2 flex flex-wrap gap-1.5">
        ${keywords.length ? keywords.map(keyword => `<span class="rounded-lg bg-primary-container/70 px-2 py-1 text-[11px] font-bold text-primary">${escHtml(keyword)}</span>`).join('') : '<span class="text-on-surface-variant/60">暂无关键词</span>'}
      </div>
    </div>
    <div class="mt-3 rounded-lg bg-surface-container-low px-3 py-2">
      <div class="font-semibold text-on-surface">父级上下文</div>
      <div class="mt-1 text-[11px] leading-relaxed text-on-surface-variant/70">${escHtml(file.file_name || '')} · ${selected?.page_num ? `第 ${selected.page_num} 页` : '当前片段'} · ${escHtml(selected?.section || '正文')}</div>
    </div>
    <div class="mt-3 flex flex-col gap-2">
      <button onclick="savePreviewChunkAsNote()" class="toolbar-btn" title="保存片段为笔记" aria-label="保存片段为笔记">
        <span class="material-symbols-outlined" style="font-size:15px">note_add</span>
        保存片段
      </button>
      <button onclick="askWithLibraryFile(${file.id})" class="toolbar-btn" title="用这个文件提问" aria-label="用这个文件提问">
        <span class="material-symbols-outlined" style="font-size:15px">chat_bubble</span>
        用它提问
      </button>
      <button onclick="openSourceOriginal()" class="toolbar-btn" title="打开原文" aria-label="打开原文">
        <span class="material-symbols-outlined" style="font-size:15px">open_in_new</span>
        打开原文
      </button>
    </div>`;
  renderLocalIcons(document.getElementById('view-source'));
}

function selectSourcePreviewChunk(index) {
  sourcePreviewState.selectedIndex = index;
  renderSourcePreview();
}

function openSourceOriginal() {
  const file = sourcePreviewState.file;
  if (file?.id) openFilePreview(file.id);
}

async function savePreviewChunkAsNote() {
  const file = sourcePreviewState.file;
  const chunk = sourcePreviewState.chunks[sourcePreviewState.selectedIndex];
  if (!file || !chunk) return;
  const text = sourceChunkText(chunk);
  if (!text.trim()) return;
  try {
    const r = await fetch(`${API}/api/notes`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        title: `来源片段 - ${file.file_name}`,
        content: `# ${file.file_name}\n\n位置：${chunk.section || ''} ${chunk.page_num ? `第 ${chunk.page_num} 页` : ''}\n\n> ${text}`,
        collection: ['Saved', 'Sources'].join(' '),
        user_tags: ['source', 'citation'],
      }),
    });
    if (!r.ok) throw new Error(await r.text());
    const detail = document.getElementById('source-detail-panel');
    if (detail) detail.insertAdjacentHTML('afterbegin', '<div class="mb-2 rounded-lg bg-tertiary-container px-3 py-2 text-[11px] font-bold text-on-tertiary-container">已保存为笔记。</div>');
    setScanButtonState('queued');
    startQueuePolling();
  } catch (e) {
    const detail = document.getElementById('source-detail-panel');
    if (detail) detail.insertAdjacentHTML('afterbegin', `<div class="mb-2 rounded-lg bg-error/10 px-3 py-2 text-[11px] font-bold text-error">保存失败：${escHtml(e.message)}</div>`);
  }
}

async function loadKnowledgeOverview(fileId = null) {
  const el = document.getElementById('library-knowledge-overview');
  if (!el) return;
  const query = fileId ? `?file_id=${encodeURIComponent(fileId)}` : '';
  try {
    const data = await fetch(`${API}/api/knowledge/overview${query}`).then(r => {
      if (!r.ok) throw new Error('知识视图读取失败');
      return r.json();
    });
    el.innerHTML = knowledgeOverviewMarkup(data);
    renderLocalIcons(el);
  } catch (e) {
    const message = userFacingErrorMessage(e.message, '知识视图暂时无法读取。');
    el.innerHTML = `<h3 class="panel-title">知识视图</h3><div class="mt-3 text-xs text-error">读取失败：${escHtml(message)}</div>`;
  }
}

function knowledgeOverviewMarkup(data) {
  const topics = data?.topics || [];
  const similar = data?.similar_documents || [];
  const cards = data?.knowledge_cards || [];
  const feedback = data?.feedback || {};
  const backlinks = data?.backlinks || [];
  const outbound = data?.outbound_links || [];
  const graph = data?.knowledge_graph || { nodes: [], edges: [], stats: {} };
  return `
    <h3 class="panel-title">知识视图</h3>
    <div class="mt-3 grid grid-cols-3 gap-2 text-xs">
      <div class="rounded-lg bg-surface-container-low px-3 py-2">
        <div class="panel-muted">主题</div>
        <div class="mt-1 font-bold text-on-surface">${topics.length}</div>
      </div>
      <div class="rounded-lg bg-surface-container-low px-3 py-2">
        <div class="panel-muted">相似资料</div>
        <div class="mt-1 font-bold text-on-surface">${similar.length}</div>
      </div>
      <div class="rounded-lg bg-surface-container-low px-3 py-2">
        <div class="panel-muted">知识卡片</div>
        <div class="mt-1 font-bold text-on-surface">${cards.length}</div>
      </div>
      <div class="rounded-lg bg-surface-container-low px-3 py-2">
        <div class="panel-muted">回答反馈</div>
        <div class="mt-1 font-bold text-on-surface">${Number(feedback.total || 0)}</div>
      </div>
      <div class="rounded-lg bg-surface-container-low px-3 py-2">
        <div class="panel-muted">反向关联</div>
        <div class="mt-1 font-bold text-on-surface">${backlinks.length}</div>
      </div>
      <div class="rounded-lg bg-surface-container-low px-3 py-2">
        <div class="panel-muted">引用来源</div>
        <div class="mt-1 font-bold text-on-surface">${outbound.length}</div>
      </div>
    </div>
    <div class="mt-4">
      <div class="text-[11px] font-bold text-on-surface-variant/60 mb-2">关系图谱</div>
      ${knowledgeGraphMarkup(graph)}
    </div>
    <div class="mt-4">
      <div class="text-[11px] font-bold text-on-surface-variant/60 mb-2">回答反馈</div>
      <div class="rounded-lg bg-surface-container-low px-3 py-3 text-xs text-on-surface-variant">
        ${feedbackSummaryMarkup(feedback)}
      </div>
    </div>
    <div class="mt-4">
      <div class="text-[11px] font-bold text-on-surface-variant/60 mb-2">反向关联</div>
      ${backlinks.length ? backlinks.slice(0, 3).map(link => knowledgeLinkMarkup(link, '引用了当前资料')).join('') : '<div class="rounded-lg bg-surface-container-low px-3 py-3 text-xs text-on-surface-variant">还没有笔记引用当前资料。</div>'}
    </div>
    <div class="mt-4">
      <div class="text-[11px] font-bold text-on-surface-variant/60 mb-2">引用来源</div>
      ${outbound.length ? outbound.slice(0, 3).map(link => knowledgeLinkMarkup(link, '当前笔记引用')).join('') : '<div class="rounded-lg bg-surface-container-low px-3 py-3 text-xs text-on-surface-variant">当前资料还没有保存来源关联。</div>'}
    </div>
    <div class="mt-4">
      <div class="text-[11px] font-bold text-on-surface-variant/60 mb-2">主题视图</div>
      ${topics.length ? topics.slice(0, 3).map(topic => `
        <button onclick="openFilePreview(${Number((topic.files || [])[0]?.id || 0)})" class="mb-2 w-full text-left rounded-lg bg-surface-container-low px-3 py-2 hover:bg-surface-container transition-colors">
          <div class="flex items-center justify-between gap-2">
            <span class="font-semibold text-on-surface">${escHtml(topic.title)}</span>
            <span class="text-[11px] text-on-surface-variant/60">${topic.file_count} 个文件</span>
          </div>
          <div class="mt-1 text-[11px] text-on-surface-variant/65 line-clamp-1">${escHtml((topic.keywords || []).join(' · ') || '来自已入库内容')}</div>
        </button>`).join('') : '<div class="rounded-lg bg-surface-container-low px-3 py-3 text-xs text-on-surface-variant">资料不足，暂时没有主题。</div>'}
    </div>
    <div class="mt-4">
      <div class="text-[11px] font-bold text-on-surface-variant/60 mb-2">相似资料</div>
      ${similar.length ? similar.slice(0, 3).map(item => similarDocumentMarkup(item)).join('') : '<div class="rounded-lg bg-surface-container-low px-3 py-3 text-xs text-on-surface-variant">还没有发现明显相似资料。</div>'}
    </div>
    <div class="mt-4">
      <div class="text-[11px] font-bold text-on-surface-variant/60 mb-2">知识卡片</div>
      ${cards.length ? cards.slice(0, 3).map(card => knowledgeCardMarkup(card)).join('') : '<div class="rounded-lg bg-surface-container-low px-3 py-3 text-xs text-on-surface-variant">完成入库后会生成可复用卡片。</div>'}
    </div>`;
}

function knowledgeGraphMarkup(graph) {
  const nodes = graph?.nodes || [];
  const edges = graph?.edges || [];
  if (!nodes.length || !edges.length) {
    return '<div class="rounded-lg bg-surface-container-low px-3 py-3 text-xs text-on-surface-variant">还没有足够关系生成图谱。</div>';
  }
  const byId = Object.fromEntries(nodes.map(node => [node.id, node]));
  const labels = {
    topic_file: '主题关联',
    similar: '内容相似',
    card_source: '卡片来源',
    backlink: '反向引用',
    source_link: '引用来源',
  };
  return `
    <div class="rounded-lg bg-surface-container-low px-3 py-3">
      <div class="mb-3 flex items-center justify-between gap-2 text-xs">
        <span class="font-semibold text-on-surface">${Number(graph?.stats?.nodes || nodes.length)} 个对象</span>
        <span class="text-on-surface-variant/60">${Number(graph?.stats?.edges || edges.length)} 条关系</span>
      </div>
      <div class="flex flex-col gap-2">
        ${edges.slice(0, 5).map(edge => {
          const source = byId[edge.source] || {};
          const target = byId[edge.target] || {};
          return `<div class="rounded-lg bg-surface-container-lowest px-3 py-2 text-xs">
            <div class="flex items-center justify-between gap-2">
              <span class="font-semibold text-on-surface line-clamp-1">${escHtml(source.label || '知识对象')}</span>
              <span class="text-[10px] font-bold text-primary">${escHtml(labels[edge.type] || '关联')}</span>
            </div>
            <div class="mt-1 text-[11px] text-on-surface-variant/65 line-clamp-1">连接到：${escHtml(target.label || '知识对象')}</div>
          </div>`;
        }).join('')}
      </div>
    </div>`;
}

function feedbackSummaryMarkup(feedback) {
  const total = Number(feedback?.total || 0);
  if (!total) return '还没有回答反馈。';
  const useful = Number(feedback.useful || 0);
  const notUseful = Number(feedback.not_useful || 0);
  const usefulRate = Math.round(Number(feedback.useful_rate || 0) * 100);
  return `有用 ${useful} 次 · 需要改进 ${notUseful} 次 · 有用率 ${usefulRate}%`;
}

function knowledgeLinkMarkup(link, label) {
  const file = link.file || {};
  return `
    <button onclick="openFilePreview(${Number(file.id || 0)})" class="mb-2 w-full text-left rounded-lg bg-surface-container-low px-3 py-2 hover:bg-surface-container transition-colors">
      <div class="flex items-center justify-between gap-2">
        <span class="font-semibold text-on-surface line-clamp-1">${escHtml(file.file_name || '关联资料')}</span>
        <span class="text-[10px] bg-primary/10 text-primary px-1.5 py-0.5 rounded font-bold whitespace-nowrap">${escHtml(label)}</span>
      </div>
      <div class="mt-1 text-[11px] text-on-surface-variant/65">${escHtml(collectionLabel(file.collection))}</div>
    </button>`;
}

function similarDocumentMarkup(item) {
  const files = item.files || [];
  const target = files[1] || files[0] || {};
  return `
    <button onclick="openFilePreview(${Number(target.id || 0)})" class="mb-2 w-full text-left rounded-lg bg-surface-container-low px-3 py-2 hover:bg-surface-container transition-colors">
      <div class="font-semibold text-on-surface line-clamp-1">${files.map(file => file.file_name).join(' ↔ ')}</div>
      <div class="mt-1 text-[11px] text-on-surface-variant/65 line-clamp-1">共同线索：${escHtml((item.shared_terms || []).join(' · ') || '内容相近')}</div>
    </button>`;
}

function knowledgeCardMarkup(card) {
  const source = card.source_file || {};
  return `
    <button onclick="openFilePreview(${Number(source.id || 0)})" class="mb-2 w-full text-left rounded-lg bg-surface-container-low px-3 py-2 hover:bg-surface-container transition-colors">
      <div class="font-semibold text-on-surface line-clamp-1">${escHtml(card.title || source.file_name || '知识卡片')}</div>
      <div class="mt-1 text-[11px] text-on-surface-variant/70 line-clamp-2">${escHtml(card.summary || '暂无摘要')}</div>
      <div class="mt-2 text-[10px] text-on-surface-variant/55">${escHtml(source.file_name || '')} · ${escHtml(collectionLabel(source.collection))}</div>
    </button>`;
}

async function openSourceReview(fileId) {
  const requestId = (librarySourceReview.requestId || 0) + 1;
  librarySourceReview = {
    fileId,
    chunks: [],
    status: 'loading',
    error: '',
    requestId,
    file: libraryFiles.find(item => item.id === fileId) || null,
  };
  renderSourceReviewPanel();
  try {
    const data = await fetch(`${API}/api/file/${fileId}/chunks?max_text_chars=1200`).then(r => {
      if (!r.ok) throw new Error('引用片段读取失败');
      return r.json();
    });
    if (librarySourceReview.requestId !== requestId || librarySourceReview.fileId !== fileId) return;
    const chunks = data.chunks || [];
    librarySourceReview = {
      fileId,
      chunks,
      status: chunks.length ? 'ready' : 'empty',
      error: '',
      requestId,
      file: data.file || libraryFiles.find(item => item.id === fileId) || null,
    };
    renderSourceReviewPanel();
  } catch (e) {
    if (librarySourceReview.requestId !== requestId || librarySourceReview.fileId !== fileId) return;
    librarySourceReview = {
      fileId,
      chunks: [],
      status: 'error',
      error: userFacingErrorMessage(e.message, '引用片段暂时无法读取。'),
      requestId,
      file: libraryFiles.find(item => item.id === fileId) || null,
    };
    renderSourceReviewPanel();
  }
}

function renderSourceReviewPanel() {
  const panel = document.getElementById('library-source-review');
  if (!panel) return;
  panel.innerHTML = sourceReviewBodyMarkup(activeLibraryFileId);
  renderLocalIcons(panel);
}

function sourceReviewBodyMarkup(fileId) {
  if (librarySourceReview.fileId !== fileId) {
    return '打开引用片段后，可查看切块、来源位置，并保存片段为笔记。';
  }
  if (librarySourceReview.status === 'loading') {
    return '<span class="spinner"></span><span class="ml-2">正在读取引用片段…</span>';
  }
  if (librarySourceReview.status === 'empty') {
    return '这个文件还没有可预览的引用片段。请确认文件已完成入库。';
  }
  if (librarySourceReview.status === 'error') {
    return `<span class="text-error">读取失败：${escHtml(librarySourceReview.error || '引用片段读取失败')}</span>`;
  }
  if (librarySourceReview.status !== 'ready') {
    return '打开引用片段后，可查看切块、来源位置，并保存片段为笔记。';
  }
  const chunks = librarySourceReview.chunks || [];
  const file = librarySourceReview.file || libraryFiles.find(item => item.id === fileId) || {};
  return `
      <div class="mb-3 rounded-lg bg-surface-container-lowest px-3 py-2">
        <div class="font-semibold text-on-surface">为什么引用这些片段</div>
        <div class="mt-1 text-[11px] leading-relaxed text-on-surface-variant/70">这些内容来自已经入库的真实片段，保留了页面、章节和文本预览，可用于核查回答证据。</div>
      </div>
      <div class="flex flex-col gap-2">
        ${chunks.slice(0, 5).map((chunk, idx) => sourceChunkMarkup(chunk, idx, file)).join('')}
      </div>`;
}

function sourceChunkMarkup(chunk, idx, file) {
  const page = chunk.page_num ? `第 ${chunk.page_num} 页` : '无页码';
  const section = chunk.section || chunk.chunk_type || '片段';
  const preview = sourceChunkText(chunk);
  return `
    <div class="rounded-lg bg-surface-container-lowest px-3 py-3">
      <div class="flex items-center justify-between gap-2">
        <div class="min-w-0">
          <div class="font-semibold text-on-surface line-clamp-1">${escHtml(section)}</div>
          <div class="text-[11px] text-on-surface-variant/60">${escHtml(page)} · ${chunk.char_count || 0} 字</div>
        </div>
        <button onclick="saveSourceChunkAsNote(${idx})" class="icon-button !w-8 !h-8" title="保存片段为笔记" aria-label="保存片段为笔记">
          <span class="material-symbols-outlined" style="font-size:15px">note_add</span>
        </button>
      </div>
      <p class="mt-2 text-[11px] leading-relaxed text-on-surface-variant line-clamp-2">${escHtml(preview || '暂无文本预览')}</p>
      <button onclick="openSourcePreview(${file.id}, ${chunk.page_num || 1})" class="mt-2 text-[11px] font-bold text-primary hover:underline">打开来源预览</button>
    </div>`;
}

async function saveSourceChunkAsNote(index) {
  const chunk = librarySourceReview.chunks[index];
  const file = libraryFiles.find(item => item.id === librarySourceReview.fileId);
  if (!chunk || !file) return;
  const text = sourceChunkText(chunk);
  if (!text.trim()) return;
  try {
    const r = await fetch(`${API}/api/notes`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        title: `来源片段 - ${file.file_name}`,
        content: `# ${file.file_name}\n\n来源：${file.file_name}\n\n位置：${chunk.section || ''} ${chunk.page_num ? `第 ${chunk.page_num} 页` : ''}\n\n> ${text}`,
        collection: ['Saved', 'Sources'].join(' '),
        user_tags: ['source', 'citation'],
      }),
    });
    if (!r.ok) throw new Error(await responseUserMessage(r, '保存片段失败，请稍后再试。'));
    const panel = document.getElementById('library-source-review');
    if (panel) {
      panel.insertAdjacentHTML('afterbegin', '<div class="mb-2 rounded-lg bg-tertiary-container px-3 py-2 text-[11px] font-bold text-on-tertiary-container">已保存为笔记，并加入入库队列。</div>');
    }
    setScanButtonState('queued');
    startQueuePolling();
  } catch (e) {
    const panel = document.getElementById('library-source-review');
    const message = userFacingErrorMessage(e.message, '保存片段失败，请稍后再试。');
    if (panel) panel.insertAdjacentHTML('afterbegin', `<div class="mb-2 rounded-lg bg-error/10 px-3 py-2 text-[11px] font-bold text-error">保存失败：${escHtml(message)}</div>`);
  }
}

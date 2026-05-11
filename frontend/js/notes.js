// ── Notes workspace ──
function insertMarkdownToken(prefix, suffix = '') {
  const input = document.getElementById('notes-content-input');
  if (!input) return;
  const start = input.selectionStart || 0;
  const end = input.selectionEnd || 0;
  const selected = input.value.slice(start, end);
  const before = input.value.slice(0, start);
  const after = input.value.slice(end);
  const needsLinePrefix = prefix.endsWith(' ') && before && !before.endsWith('\n');
  const insertPrefix = needsLinePrefix ? `\n${prefix}` : prefix;
  const nextText = `${insertPrefix}${selected || ''}${suffix}`;
  input.value = `${before}${nextText}${after}`;
  const cursor = start + insertPrefix.length + (selected ? selected.length : 0);
  input.focus();
  input.setSelectionRange(cursor, cursor);
  updateNotesEditorStats();
}

function updateNotesEditorStats() {
  const title = document.getElementById('notes-title-input');
  const content = document.getElementById('notes-content-input');
  const words = document.getElementById('notes-word-count');
  const titleCount = document.getElementById('notes-title-count');
  if (words && content) {
    const count = (content.value || '').trim().replace(/\s+/g, '').length;
    words.textContent = `${count} 字`;
  }
  if (titleCount && title) titleCount.textContent = `${(title.value || '').length}/200`;
}

function setKnowledgeOutputType(type) {
  const select = document.getElementById('knowledge-output-type');
  if (!select) return;
  select.value = type;
  syncKnowledgeOutputCards();
}

function syncKnowledgeOutputCards() {
  const select = document.getElementById('knowledge-output-type');
  const value = select?.value || 'summary';
  document.querySelectorAll('.knowledge-output-card').forEach(card => {
    const active = card.dataset.outputType === value;
    card.classList.toggle('bg-primary-container', active);
    card.classList.toggle('text-primary', active);
    card.classList.toggle('bg-surface-container-low', !active);
  });
}

async function createNoteFromNotesView() {
  const btn = document.getElementById('notes-submit-btn');
  const status = document.getElementById('notes-status');
  const title = document.getElementById('notes-title-input').value.trim() || 'Untitled Note';
  const content = document.getElementById('notes-content-input').value;
  const collection = document.getElementById('notes-collection-input').value.trim() || 'Notes';
  const userTags = parseTagsInput(document.getElementById('notes-tags-input').value);
  if (!content.trim()) {
    status.textContent = '请先输入笔记内容';
    status.classList.add('text-error');
    return;
  }
  btn.disabled = true;
  status.classList.remove('text-error');
  status.textContent = '正在保存…';
  try {
    const r = await fetch(`${API}/api/notes`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        title,
        content,
        collection,
        user_tags: userTags.length ? userTags : ['note'],
      }),
    });
    if (!r.ok) throw new Error(await r.text());
    document.getElementById('notes-title-input').value = '';
    document.getElementById('notes-content-input').value = '';
    document.getElementById('notes-collection-input').value = '';
    document.getElementById('notes-tags-input').value = '';
    updateNotesEditorStats();
    status.textContent = '已加入入库队列';
    setScanButtonState('queued');
    startQueuePolling();
    await refreshNotesView();
  } catch (e) {
    status.textContent = `保存失败：${e.message}`;
    status.classList.add('text-error');
  } finally {
    btn.disabled = false;
  }
}

async function importUrlFromNotesView() {
  const btn = document.getElementById('notes-url-submit-btn');
  const status = document.getElementById('notes-url-status');
  const url = document.getElementById('notes-url-input').value.trim();
  const title = document.getElementById('notes-url-title-input').value.trim();
  const collection = document.getElementById('notes-url-collection-input').value.trim() || 'Web Imports';
  const userTags = parseTagsInput(document.getElementById('notes-url-tags-input').value);
  if (!url) {
    status.textContent = '请输入网页地址';
    status.classList.add('text-error');
    return;
  }
  const extractMain = document.getElementById('notes-url-extract-main')?.checked;
  const queueIndex = document.getElementById('notes-url-queue-index')?.checked;
  if (!extractMain || !queueIndex) {
    status.textContent = '网页导入当前需要提取正文并加入资料库';
    status.classList.add('text-error');
    return;
  }
  btn.disabled = true;
  status.classList.remove('text-error');
  status.textContent = '正在导入…';
  try {
    const r = await fetch(`${API}/api/import/url`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        url,
        title: title || null,
        collection,
        user_tags: userTags.length ? userTags : ['web'],
      }),
    });
    if (!r.ok) throw new Error(await r.text());
    document.getElementById('notes-url-input').value = '';
    document.getElementById('notes-url-title-input').value = '';
    document.getElementById('notes-url-collection-input').value = '';
    document.getElementById('notes-url-tags-input').value = '';
    status.textContent = '已加入入库队列';
    setScanButtonState('queued');
    startQueuePolling();
    await refreshNotesView();
  } catch (e) {
    status.textContent = `导入失败：${e.message}`;
    status.classList.add('text-error');
  } finally {
    btn.disabled = false;
  }
}

function openKnowledgeFromSelectedFiles() {
  if (selectedFileIds.size === 0) return;
  knowledgeSourceFileIds = [...selectedFileIds];
  switchView('notes');
  const title = document.getElementById('knowledge-title-input');
  if (title && !title.value.trim()) title.value = `选中文件知识产物`;
  updateKnowledgeSourceFilesLabel();
  document.getElementById('knowledge-output-panel')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  document.getElementById('knowledge-source-input')?.focus();
}

function clearKnowledgeSourceFiles() {
  knowledgeSourceFileIds = [];
  updateKnowledgeSourceFilesLabel();
}

function updateKnowledgeSourceFilesLabel() {
  const label = document.getElementById('knowledge-source-files-label');
  const clearBtn = document.getElementById('knowledge-clear-files-btn');
  if (!label || !clearBtn) return;
  if (knowledgeSourceFileIds.length) {
    label.textContent = `将使用资料库中选中的 ${knowledgeSourceFileIds.length} 个文件`;
    clearBtn.classList.remove('hidden');
    clearBtn.classList.add('inline-flex');
  } else {
    label.textContent = '可粘贴资料，也可从资料库选中文件带入';
    clearBtn.classList.add('hidden');
    clearBtn.classList.remove('inline-flex');
  }
}

async function createKnowledgeOutputFromNotesView() {
  const btn = document.getElementById('knowledge-submit-btn');
  const status = document.getElementById('knowledge-status');
  const outputType = document.getElementById('knowledge-output-type').value;
  const title = document.getElementById('knowledge-title-input').value.trim();
  const sourceText = document.getElementById('knowledge-source-input').value;
  const collection = document.getElementById('knowledge-collection-input').value.trim() || knowledgeOutputCollectionName();
  const userTags = parseTagsInput(document.getElementById('knowledge-tags-input').value);
  if (!sourceText.trim() && knowledgeSourceFileIds.length === 0) {
    status.textContent = '请先输入资料，或从资料库选中文件';
    status.classList.add('text-error');
    return;
  }
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span><span class="ml-1.5">生成中…</span>';
  status.classList.remove('text-error');
  status.textContent = '正在生成并保存…';
  try {
    const r = await fetch(`${API}/api/knowledge-output`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        output_type: outputType,
        title: title || null,
        source_text: sourceText,
        file_ids: knowledgeSourceFileIds,
        collection,
        user_tags: userTags.length ? userTags : null,
      }),
    });
    if (!r.ok) throw new Error(await r.text());
    const data = await r.json();
    document.getElementById('knowledge-title-input').value = '';
    document.getElementById('knowledge-source-input').value = '';
    document.getElementById('knowledge-collection-input').value = '';
    document.getElementById('knowledge-tags-input').value = '';
    knowledgeSourceFileIds = [];
    updateKnowledgeSourceFilesLabel();
    status.textContent = `已加入入库队列：${data.file?.file_name || '知识产物'}`;
    setScanButtonState('queued');
    startQueuePolling();
    await refreshNotesView();
  } catch (e) {
    status.textContent = `生成失败：${e.message}`;
    status.classList.add('text-error');
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<span class="material-symbols-outlined" style="font-size:15px">auto_awesome</span><span class="ml-1">生成并入库</span>';
  }
}

async function refreshNotesView() {
  const list = document.getElementById('notes-list');
  const count = document.getElementById('notes-count');
  if (!list) return;
  try {
    const [files, meta, history, queue] = await Promise.all([
      fetch(`${API}/api/files?status=done`).then(r => r.json()),
      fetch(`${API}/api/library/meta`).then(r => r.json()).catch(() => libraryMeta),
      fetch(`${API}/api/history?limit=10`).then(r => r.json()).catch(() => []),
      fetch(`${API}/api/queue`).then(r => r.json()).catch(() => null),
    ]);
    historyItems = Array.isArray(history) ? history : historyItems;
    libraryMeta = meta || libraryMeta;
    updateLibraryControls();
    updateKnowledgeSourceFilesLabel();
    syncKnowledgeOutputCards();
    updateNotesEditorStats();
    const savedAnswersCollection = ['Saved', 'Answers'].join(' ');
    const noteCollections = new Set(['Notes', 'Web Imports', savedAnswersCollection, knowledgeOutputCollectionName()]);
    const notes = files
      .filter(file => noteCollections.has(file.collection || ''))
      .slice(0, 12);
    const recentCaptures = notes.length ? notes : files.slice(0, 12);
    const knowledge = files.filter(file => file.collection === knowledgeOutputCollectionName()).slice(0, 4);
    const savedAnswers = files.filter(file => file.collection === savedAnswersCollection).slice(0, 4);
    const displayKnowledge = knowledge.length ? knowledge : notes.slice(0, 4);
    if (count) count.textContent = `${displayKnowledge.length} 个`;
    list.innerHTML = displayKnowledge.length ? displayKnowledge.map(file => `
      <button onclick="openFilePreview(${file.id})" class="text-left rounded-lg bg-surface-container-low px-3 py-3 hover:bg-surface-container active:scale-95 transition-all">
        <div class="flex items-start justify-between gap-3">
          <div class="min-w-0">
            <div class="text-xs font-semibold text-on-surface line-clamp-1">${escHtml(file.file_name)}</div>
            <div class="text-[11px] text-on-surface-variant/60 mt-1 line-clamp-1">${escHtml(collectionLabel(file.collection))}</div>
          </div>
          <span class="material-symbols-outlined text-primary" style="font-size:16px">open_in_new</span>
        </div>
      </button>`).join('') : '<div class="rounded-lg bg-surface-container-low px-3 py-8 text-center">还没有知识产物。</div>';

    const savedList = document.getElementById('saved-answers-list');
    if (savedList) {
      const answerItems = savedAnswers.length
        ? savedAnswers.map(file => ({
            title: file.file_name,
            meta: `${collectionLabel(file.collection)} · ${(file.updated_at || '').slice(0, 10)}`,
            action: `openFilePreview(${file.id})`,
          }))
        : (historyItems || []).slice(0, 3).map((item, idx) => ({
            title: item.question || '未命名回答',
            meta: `来自对话 · ${(item.created_at || '').slice(0, 10)}`,
            action: `replayHistory(${idx})`,
          }));
      savedList.innerHTML = answerItems.length ? answerItems.map(item => `
        <button onclick="${item.action}" class="w-full text-left rounded-lg bg-surface-container-low px-3 py-2 hover:bg-surface-container transition-colors">
          <div class="font-semibold text-on-surface line-clamp-1">${escHtml(item.title)}</div>
          <div class="mt-0.5 text-[11px] text-on-surface-variant/60">${escHtml(item.meta)}</div>
        </button>`).join('') : '<div class="rounded-lg bg-surface-container-low px-3 py-3">还没有保存回答。</div>';
    }

    renderNotesTimeline(queue, recentCaptures);
    renderNotesRecentTable(recentCaptures);
  } catch (e) {
    list.innerHTML = `<div class="text-error text-sm">笔记列表加载失败：${escHtml(e.message)}</div>`;
  }
}

function renderNotesTimeline(queue, notes) {
  const el = document.getElementById('notes-processing-timeline');
  if (!el) return;
  if (queue?.queue_size > 0 || queue?.processing) {
    const stage = queueStageLabel(queue.progress?.stage || 'processing');
    el.innerHTML = `
      <div class="flex gap-3">
        <span class="source-timeline-dot mt-1"></span>
        <div class="min-w-0">
          <div class="font-semibold text-on-surface">${queue.processing ? `正在整理：${escHtml(queue.processing)}` : '等待整理'}</div>
          <div class="mt-1 text-[11px] text-on-surface-variant/60">${escHtml(stage)} · 队列 ${queue.queue_size || 0} 个</div>
        </div>
      </div>`;
    return;
  }
  const recent = (notes || []).slice(0, 4);
  el.innerHTML = recent.length ? recent.map(file => `
    <div class="flex gap-3">
      <span class="source-timeline-dot mt-1"></span>
      <div class="min-w-0">
        <div class="font-semibold text-on-surface line-clamp-1">${escHtml(file.file_name)}</div>
        <div class="mt-1 text-[11px] text-on-surface-variant/60">${escHtml(collectionLabel(file.collection))} · ${fileStatusLabel(file.status)}</div>
      </div>
    </div>`).join('') : '<div class="rounded-lg bg-surface-container-low px-3 py-3">暂无处理记录。</div>';
}

function queueStageLabel(stage) {
  return {
    processing: '整理中',
    parsing: '解析资料',
    embedding: '建立索引',
    storing: '保存记录',
    done: '已完成',
  }[stage] || '整理中';
}

function renderNotesRecentTable(notes) {
  const table = document.getElementById('notes-recent-table');
  const count = document.getElementById('notes-recent-count');
  if (!table) return;
  const rows = (notes || []).slice(0, 6);
  if (count) count.textContent = `${rows.length} 条`;
  if (!rows.length) {
    table.innerHTML = '<tr><td colspan="6" class="py-8 text-center text-on-surface-variant/60">暂无最近采集。</td></tr>';
    return;
  }
  table.innerHTML = rows.map(file => `
    <tr class="hover:bg-surface-container/50">
      <td class="py-2 pr-3">${fileTypeLabel(file.file_name, file.is_scanned)}</td>
      <td class="py-2 pr-3 font-semibold text-on-surface line-clamp-1">${escHtml(file.file_name)}</td>
      <td class="py-2 pr-3">${escHtml(collectionLabel(file.collection))}</td>
      <td class="py-2 pr-3">${tagPills(Array.isArray(file.user_tags) ? file.user_tags : [])}</td>
      <td class="py-2 pr-3">${statusBadge(file.status)}</td>
      <td class="py-2 text-on-surface-variant/60">${(file.updated_at || '').slice(0, 16) || '-'}</td>
    </tr>`).join('');
  renderLocalIcons(table);
}

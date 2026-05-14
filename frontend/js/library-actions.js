async function loadLibraryFileActivity(file) {
  const el = document.getElementById('library-recent-citations');
  if (!el || !file) return;
  try {
    const history = await fetch(`${API}/api/history?limit=50`).then(r => r.json());
    const matches = history.filter(item => (item.citations || []).some(c =>
      c.file_path === file.file_path || c.file_name === file.file_name
    )).slice(0, 3);
    if (!matches.length) {
      el.innerHTML = '<div class="rounded-lg bg-surface-container-low px-3 py-3">暂无最近引用。</div>';
      return;
    }
    el.innerHTML = matches.map(item => `
      <button data-question="${escHtml(item.question || '')}" onclick="replayHistoryFromLibrary(this)" class="w-full text-left rounded-lg bg-surface-container-low px-3 py-3 hover:bg-surface-container transition-colors">
        <div class="font-semibold text-on-surface line-clamp-1">${escHtml(item.question || '未命名问题')}</div>
        <div class="mt-1 text-[11px] text-on-surface-variant/60">${(item.created_at || '').slice(0,16)}</div>
      </button>`).join('');
  } catch (e) {
    const message = userFacingErrorMessage(e.message, '最近引用暂时无法读取。');
    el.innerHTML = `<span class="text-error">最近引用读取失败：${escHtml(message)}</span>`;
  }
}

async function askWithLibraryFile(fileId) {
  const file = libraryFiles.find(item => item.id === fileId) || (sourcePreviewState.file?.id === fileId ? sourcePreviewState.file : null);
  switchView('chat');
  await loadQueryScopeOptions(true);
  const mode = document.getElementById('query-scope-mode');
  const select = document.getElementById('query-scope-file');
  if (mode) mode.value = 'file';
  updateQueryScopeControls();
  if (select) select.value = String(fileId);
  updateQueryScopeControls();
  const input = document.getElementById('input');
  input.value = file ? `请基于《${file.file_name}》总结核心要点` : '';
  autoResize(input);
  input.focus();
}

function replayHistoryFromLibrary(button) {
  const question = button?.dataset?.question || '';
  switchView('chat');
  const input = document.getElementById('input');
  input.value = question;
  autoResize(input);
  input.focus();
}

function selectSingleAndRebuild(fileId) {
  selectedFileIds = new Set([fileId]);
  updateSummarizeBar();
  rebuildSelected();
}

function selectSingleAndOpenKnowledge(fileId) {
  selectedFileIds = new Set([fileId]);
  updateSummarizeBar();
  openKnowledgeFromSelectedFiles();
}

function collectionBadge(collection) {
  return `<span class="inline-flex items-center gap-1 px-2 py-1 rounded-lg bg-surface-container-high text-on-surface-variant">
    <span class="material-symbols-outlined" style="font-size:12px">inventory_2</span>${escHtml(collectionLabel(collection))}
  </span>`;
}

function tagPills(tags) {
  if (!tags.length) return '<span class="text-on-surface-variant/35">-</span>';
  return `<div class="flex flex-wrap gap-1">${tags.slice(0, 3).map(tag => `
    <span class="inline-flex items-center px-2 py-1 rounded-lg bg-primary-container/70 text-primary text-[11px] font-medium">${escHtml(tag)}</span>
  `).join('')}${tags.length > 3 ? `<span class="text-[11px] text-on-surface-variant/50">+${tags.length - 3}</span>` : ''}</div>`;
}

function statusBadge(status) {
  const map = {
    done: 'bg-tertiary-container text-on-tertiary-container',
    processing: 'bg-surface-container-high text-on-surface-variant',
    error: 'bg-surface-container-high text-error',
    pending: 'bg-surface-container text-on-surface-variant/60',
  };
  const icons = {
    done: 'check_circle',
    processing: 'autorenew',
    error: 'error',
    pending: 'schedule',
  };
  const cls = map[status] || map.pending;
  const icon = icons[status] || 'schedule';
  const label = fileStatusLabel(status);
  return `<span class="inline-flex items-center gap-1 px-2 py-1 rounded-full text-[11px] font-bold ${cls}">
    <span class="material-symbols-outlined" style="font-size:12px">${icon}</span>${escHtml(label)}
  </span>`;
}

function fileStatusLabel(status) {
  const labels = {
    done: '已入库',
    processing: '处理中',
    error: '失败',
    pending: '等待中',
  };
  return labels[status] || status || '等待中';
}

function toggleFileSelect(cb) {
  const id = parseInt(cb.dataset.id);
  if (cb.checked) selectedFileIds.add(id);
  else selectedFileIds.delete(id);
  updateSummarizeBar();
}

function toggleSelectAll(master) {
  document.querySelectorAll('#file-tbody input[type=checkbox]:not(:disabled)').forEach(cb => {
    cb.checked = master.checked;
    const id = parseInt(cb.dataset.id);
    if (master.checked) selectedFileIds.add(id);
    else selectedFileIds.delete(id);
  });
  updateSummarizeBar();
}

function updateSummarizeBar() {
  const bar = document.getElementById('summarize-bar');
  const count = document.getElementById('selected-count');
  if (selectedFileIds.size > 0) {
    bar.classList.remove('hidden');
    bar.classList.add('flex');
    count.textContent = `已选 ${selectedFileIds.size} 个文件`;
  } else {
    bar.classList.add('hidden');
    bar.classList.remove('flex');
  }
}

async function toggleFavorite(fileId, btn) {
  try {
    const r = await fetch(`${API}/api/favorites/${fileId}`, { method: 'POST' });
    if (!r.ok) throw new Error(await responseUserMessage(r, '收藏失败，请稍后再试。'));
    const data = await r.json();
    const icon = btn.querySelector('.material-symbols-outlined');
    if (data.favorited) {
      icon.classList.add('icon-fill', 'text-primary');
      favoritedIds.add(fileId);
      btn.title = '取消收藏';
      btn.setAttribute('aria-label', '取消收藏');
    } else {
      icon.classList.remove('icon-fill', 'text-primary');
      favoritedIds.delete(fileId);
      btn.title = '收藏';
      btn.setAttribute('aria-label', '收藏');
    }
    await refreshFiles();
  } catch {}
}

function openFilePreview(fileId) {
  const item = libraryFiles.find(file => file.id === fileId);
  const fileName = item?.file_name || '';
  const suffix = (fileName || '').split('.').pop().toLowerCase();
  const hash = suffix === 'pdf' ? '#page=1' : '';
  window.open(`${API}/api/file/${fileId}/preview${hash}`, '_blank');
}

function parseTagsInput(value) {
  return String(value || '')
    .split(/[,，#\n]/)
    .map(item => item.trim())
    .filter(Boolean);
}

async function favoriteSelected() {
  if (selectedFileIds.size === 0) return;
  const btn = document.getElementById('favorite-selected-btn');
  btn.disabled = true;
  try {
    const r = await fetch(`${API}/api/files/batch/favorite`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ file_ids: [...selectedFileIds], favorited: true }),
    });
    if (!r.ok) throw new Error(await responseUserMessage(r, '收藏失败，请稍后再试。'));
    await refreshFiles();
  } catch (e) {
    alert(userFacingErrorMessage(e.message, '批量收藏失败，请稍后再试。'));
  } finally {
    btn.disabled = false;
  }
}

async function applyBatchMetadata() {
  if (selectedFileIds.size === 0) return;
  const collection = document.getElementById('batch-collection-input').value.trim();
  const userTags = parseTagsInput(document.getElementById('batch-tags-input').value);
  if (!collection && !userTags.length) return;
  const btn = document.getElementById('apply-metadata-btn');
  btn.disabled = true;
  try {
    const r = await fetch(`${API}/api/files/batch/metadata`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        file_ids: [...selectedFileIds],
        collection: collection || null,
        user_tags: userTags.length ? userTags : null,
      }),
    });
    if (!r.ok) throw new Error(await responseUserMessage(r, '批量更新失败，请稍后再试。'));
    document.getElementById('batch-collection-input').value = '';
    document.getElementById('batch-tags-input').value = '';
    await refreshFiles();
  } catch (e) {
    alert(userFacingErrorMessage(e.message, '批量更新失败，请稍后再试。'));
  } finally {
    btn.disabled = false;
  }
}

async function rebuildSelected() {
  if (selectedFileIds.size === 0) return;
  const confirmed = await showConfirmDialog({
    title: '重新整理选中文件？',
    message: `将重新读取 ${selectedFileIds.size} 个文件，并更新资料状态。`,
    confirmText: '开始整理',
  });
  if (!confirmed) return;
  const btn = document.getElementById('rebuild-selected-btn');
  btn.disabled = true;
  try {
    const r = await fetch(`${API}/api/files/batch/rebuild`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ file_ids: [...selectedFileIds] }),
    });
    if (!r.ok) throw new Error(await responseUserMessage(r, '整理失败，请稍后再试。'));
    setScanButtonState('queued');
    pollQueueOnce();
    await refreshFiles();
  } catch (e) {
    alert(userFacingErrorMessage(e.message, '整理失败，请稍后再试。'));
  } finally {
    btn.disabled = false;
  }
}

async function summarizeSelected() {
  if (selectedFileIds.size === 0) return;
  const btn = document.getElementById('summarize-btn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span><span class="ml-1.5">生成中…</span>';
  try {
    const r = await fetch(`${API}/api/summarize`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ file_ids: [...selectedFileIds] }),
    });
    if (!r.ok) throw new Error(await responseUserMessage(r, '摘要生成失败，请稍后再试。'));
    const md = await r.text();
    const blob = new Blob([md], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'docflow-summary.md'; a.click();
    URL.revokeObjectURL(url);
  } catch (e) {
    alert(userFacingErrorMessage(e.message, '摘要生成失败，请稍后再试。'));
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<span class="material-symbols-outlined" style="font-size:15px">summarize</span><span class="ml-1">生成摘要</span>';
  }
}

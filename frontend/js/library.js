// ── Library page ──
function fileTypeLabel(fileName, isScanned) {
  const ext = (fileName || '').split('.').pop().toLowerCase();
  if (ext === 'pdf') return isScanned ? 'OCR 扫描件' : '原生 PDF';
  if (ext === 'md' || ext === 'markdown') return 'Markdown';
  if (ext === 'txt') return '文本文件';
  if (ext === 'docx') return 'Word 文档';
  if (['jpg','jpeg','png','webp','heic','heif'].includes(ext)) return '图片';
  return ext.toUpperCase() || '未知';
}

function setRefreshButtonLoading(loading) {
  const btn = document.getElementById('refresh-files-btn');
  const icon = document.getElementById('refresh-files-icon');
  const label = document.getElementById('refresh-files-label');
  if (!btn || !icon || !label) return;
  btn.disabled = loading;
  setIcon(icon, loading ? 'autorenew' : 'refresh');
  icon.classList.toggle('animate-spin', loading);
  label.textContent = loading ? '刷新中…' : '刷新列表';
}

function setScanButtonState(state) {
  const btn = document.getElementById('scan-folders-btn');
  const icon = document.getElementById('scan-folders-icon');
  const label = document.getElementById('scan-folders-label');
  if (!btn || !icon || !label) return;
  const loading = state === 'loading';
  btn.disabled = loading;
  icon.classList.toggle('animate-spin', loading);
  if (state === 'loading') {
    setIcon(icon, 'sync');
    label.textContent = '扫描中…';
  } else if (state === 'queued') {
    setIcon(icon, 'task_alt');
    label.textContent = '已加入队列';
  } else {
    setIcon(icon, 'folder_sync');
    label.textContent = '扫描文件夹';
  }
}

function currentFileQuery() {
  const params = new URLSearchParams();
  if (libraryFilters.status) params.set('status', libraryFilters.status);
  if (libraryFilters.collection) params.set('collection', libraryFilters.collection);
  if (libraryFilters.tag) params.set('tag', libraryFilters.tag);
  if (libraryFilters.favorite) params.set('favorite', 'true');
  if (libraryFilters.kind) params.set('kind', libraryFilters.kind);
  if (libraryFilters.recent) params.set('recent', 'true');
  const query = params.toString();
  return query ? `?${query}` : '';
}

function applyLibraryFilters() {
  libraryPage = 1;
  libraryFilters = {
    status: document.getElementById('library-status-filter')?.value || '',
    collection: document.getElementById('library-collection-filter')?.value || '',
    tag: document.getElementById('library-tag-filter')?.value || '',
    favorite: Boolean(document.getElementById('library-favorite-filter')?.checked),
    kind: libraryFilters.kind || '',
    recent: Boolean(libraryFilters.recent),
  };
  refreshFiles({ preserveFilters: true });
}

function setLibraryGroup(group) {
  libraryPage = 1;
  libraryFilters.kind = '';
  libraryFilters.recent = false;
  libraryFilters.favorite = false;
  const favoriteFilter = document.getElementById('library-favorite-filter');
  if (favoriteFilter) favoriteFilter.checked = false;
  if (group === 'favorites') {
    libraryFilters.favorite = true;
    if (favoriteFilter) favoriteFilter.checked = true;
  } else if (group === 'recent') {
    libraryFilters.recent = true;
  } else if (['pdf', 'markdown', 'image', 'code'].includes(group)) {
    libraryFilters.kind = group;
  }
  updateLibraryGroupControls();
  refreshFiles({ preserveFilters: true });
}

function clearLibraryFilters() {
  libraryPage = 1;
  document.getElementById('library-status-filter').value = '';
  document.getElementById('library-collection-filter').value = '';
  document.getElementById('library-tag-filter').value = '';
  document.getElementById('library-favorite-filter').checked = false;
  libraryFilters = { status: '', collection: '', tag: '', favorite: false, kind: '', recent: false };
  updateLibraryGroupControls();
  refreshFiles({ preserveFilters: true });
}

function updateLibraryControls() {
  const collectionSelect = document.getElementById('library-collection-filter');
  const tagSelect = document.getElementById('library-tag-filter');
  const collectionOptions = document.getElementById('collection-options');
  const tagOptions = document.getElementById('tag-options');
  const collectionList = document.getElementById('library-collection-list');
  const tagList = document.getElementById('library-tag-list');
  if (collectionSelect) {
    const value = collectionSelect.value;
    collectionSelect.innerHTML = '<option value="">全部集合</option>' + (libraryMeta.collections || [])
      .map(item => `<option value="${escHtml(item.name)}">${escHtml(item.name)} (${item.count})</option>`)
      .join('');
    collectionSelect.value = value;
  }
  if (tagSelect) {
    const value = tagSelect.value;
    tagSelect.innerHTML = '<option value="">全部标签</option>' + (libraryMeta.user_tags || [])
      .map(item => `<option value="${escHtml(item.name)}">${escHtml(item.name)} (${item.count})</option>`)
      .join('');
    tagSelect.value = value;
  }
  if (collectionOptions) {
    collectionOptions.innerHTML = (libraryMeta.collections || [])
      .map(item => `<option value="${escHtml(item.name)}"></option>`)
      .join('');
  }
  if (tagOptions) {
    tagOptions.innerHTML = (libraryMeta.user_tags || [])
      .map(item => `<option value="${escHtml(item.name)}"></option>`)
      .join('');
  }
  if (collectionList) {
    const items = (libraryMeta.collections || []).slice(0, 8);
    collectionList.innerHTML = items.length
      ? items.map(item => libraryFacetButton('collection', item.name, item.count)).join('')
      : '<div class="rounded-lg bg-surface-container-low px-3 py-3 text-on-surface-variant/60">暂无集合</div>';
  }
  if (tagList) {
    const items = (libraryMeta.user_tags || []).slice(0, 8);
    tagList.innerHTML = items.length
      ? items.map(item => libraryFacetButton('tag', item.name, item.count)).join('')
      : '<div class="rounded-lg bg-surface-container-low px-3 py-3 text-on-surface-variant/60">暂无标签</div>';
  }
  updateLibraryGroupControls();
  renderLocalIcons(document.getElementById('view-library'));
}

function libraryFacetButton(type, name, count) {
  const active = type === 'collection' ? libraryFilters.collection === name : libraryFilters.tag === name;
  const action = type === 'collection' ? 'setLibraryCollectionFilter' : 'setLibraryTagFilter';
  return `<button onclick="${action}(decodeURIComponent('${encodeURIComponent(name)}'))" class="w-full min-h-9 rounded-lg px-3 flex items-center justify-between gap-2 text-left transition-all active:scale-95 ${active ? 'bg-primary-container text-primary' : 'bg-surface-container-low text-on-surface-variant hover:bg-surface-container'}">
    <span class="inline-flex items-center gap-2 min-w-0">
      <span class="material-symbols-outlined" style="font-size:13px">${type === 'collection' ? 'inventory_2' : 'sell'}</span>
      <span class="truncate">${escHtml(name)}</span>
    </span>
    <span class="text-[11px] font-bold">${count || 0}</span>
  </button>`;
}

function setLibraryCollectionFilter(name) {
  libraryPage = 1;
  const select = document.getElementById('library-collection-filter');
  if (select) select.value = name;
  libraryFilters.collection = name;
  refreshFiles({ preserveFilters: true });
}

function setLibraryTagFilter(name) {
  libraryPage = 1;
  const select = document.getElementById('library-tag-filter');
  if (select) select.value = name;
  libraryFilters.tag = name;
  refreshFiles({ preserveFilters: true });
}

function currentLibraryGroup() {
  if (libraryFilters.favorite) return 'favorites';
  if (libraryFilters.recent) return 'recent';
  return libraryFilters.kind || 'all';
}

function updateLibraryGroupControls() {
  const counts = {
    all: libraryMeta.total_files || 0,
    favorites: libraryMeta.favorites || 0,
    recent: libraryMeta.recent || 0,
    pdf: libraryMeta.types?.pdf || 0,
    markdown: libraryMeta.types?.markdown || 0,
    image: libraryMeta.types?.image || 0,
    code: libraryMeta.types?.code || 0,
  };
  Object.entries(counts).forEach(([group, count]) => {
    const countEl = document.getElementById(`library-group-${group}-count`);
    if (countEl) countEl.textContent = count;
  });
  const active = currentLibraryGroup();
  document.querySelectorAll('.library-group-btn').forEach(btn => btn.classList.remove('active-nav'));
  document.getElementById(`library-group-${active}`)?.classList.add('active-nav');
}

function openLibraryWorkflow(mode) {
  libraryWorkflowMode = mode;
  const panel = document.getElementById('library-workflow-panel');
  const isUrl = mode === 'url';
  document.getElementById('workflow-title').textContent = isUrl ? '导入网页' : '新建笔记';
  document.getElementById('workflow-subtitle').textContent = isUrl
    ? '抓取网页正文，保存为本地 Markdown 并加入入库队列'
    : '写入本地监控目录并加入入库队列';
  document.getElementById('workflow-url-input').classList.toggle('hidden', !isUrl);
  document.getElementById('workflow-content-input').classList.toggle('hidden', isUrl);
  document.getElementById('workflow-status').textContent = '';
  panel.classList.remove('hidden');
  (isUrl ? document.getElementById('workflow-url-input') : document.getElementById('workflow-title-input')).focus();
}

function closeLibraryWorkflow() {
  document.getElementById('library-workflow-panel').classList.add('hidden');
}

async function runLibraryWorkflow() {
  const btn = document.getElementById('workflow-submit-btn');
  const status = document.getElementById('workflow-status');
  const title = document.getElementById('workflow-title-input').value.trim();
  const collection = document.getElementById('workflow-collection-input').value.trim();
  const userTags = parseTagsInput(document.getElementById('workflow-tags-input').value);
  const isUrl = libraryWorkflowMode === 'url';
  const payload = isUrl
    ? {
        url: document.getElementById('workflow-url-input').value.trim(),
        title: title || null,
        collection: collection || null,
        user_tags: userTags.length ? userTags : null,
      }
    : {
        title: title || 'Untitled Note',
        content: document.getElementById('workflow-content-input').value,
        collection: collection || null,
        user_tags: userTags.length ? userTags : null,
      };
  const endpoint = isUrl ? '/api/import/url' : '/api/notes';
  btn.disabled = true;
  status.classList.remove('text-error');
  status.textContent = isUrl ? '正在导入网页…' : '正在保存笔记…';
  try {
    const r = await fetch(`${API}${endpoint}`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload),
    });
    if (!r.ok) throw new Error(await r.text());
    const data = await r.json();
    status.textContent = `已加入队列：${data.file?.file_name || data.path}`;
    document.getElementById('workflow-url-input').value = '';
    document.getElementById('workflow-title-input').value = '';
    document.getElementById('workflow-content-input').value = '';
    document.getElementById('workflow-collection-input').value = '';
    document.getElementById('workflow-tags-input').value = '';
    setScanButtonState('queued');
    pollQueueOnce();
    await refreshFiles();
  } catch (e) {
    status.textContent = `失败：${e.message}`;
    status.classList.add('text-error');
  } finally {
    btn.disabled = false;
  }
}

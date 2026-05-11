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

async function refreshFiles(options = {}) {
  const showLoading = Boolean(options.showLoading);
  const requestId = ++refreshFilesRequestId;
  if (showLoading) setRefreshButtonLoading(true);
  try {
    const [files, meta, allFiles] = await Promise.all([
      fetch(`${API}/api/files${currentFileQuery()}`).then(r => r.json()),
      fetch(`${API}/api/library/meta`).then(r => r.json()).catch(() => libraryMeta),
      fetch(`${API}/api/files`).then(r => r.json()).catch(() => []),
    ]);
    if (requestId !== refreshFilesRequestId) return;
    libraryFiles = files;
    libraryMeta = normalizeLibraryMeta(meta || libraryMeta, allFiles);
    updateSidebarStorageSummary();
    if (!files.some(file => file.id === activeLibraryFileId)) {
      activeLibraryFileId = files[0]?.id || null;
    }
    updateLibraryControls();
    if (!document.getElementById('view-chat').classList.contains('hidden')) {
      loadQueryScopeOptions();
    }
    favoritedIds = new Set(files.filter(f => f.favorited).map(f => f.id));
    selectedFileIds.clear();
    updateSummarizeBar();

    const tbody = document.getElementById('file-tbody');
    const totalPages = Math.max(1, Math.ceil(files.length / libraryPageSize));
    libraryPage = Math.min(Math.max(1, libraryPage), totalPages);
    const start = (libraryPage - 1) * libraryPageSize;
    const visibleFiles = files.slice(start, start + libraryPageSize);
    document.getElementById('library-count').textContent = `显示 ${files.length} / ${libraryMeta.total_files ?? files.length} 个文件`;
    if (!files.length) {
      tbody.innerHTML = `<tr><td colspan="10" class="py-12">
        <div class="mx-auto max-w-2xl text-center">
          <div class="text-sm font-semibold text-on-surface">还没有可查看的资料</div>
          <div class="mt-1 text-xs text-on-surface-variant/60">先导入示例资料，或把自己的文件加入资料库。</div>
          <div class="mt-4 flex flex-wrap items-center justify-center gap-2">
            <button onclick="createDemoLibrary()" class="toolbar-btn toolbar-btn-primary">
              <span class="material-symbols-outlined" style="font-size:16px">auto_awesome</span>
              导入示例资料
            </button>
            <button onclick="document.getElementById('file-input')?.click()" class="toolbar-btn">
              <span class="material-symbols-outlined" style="font-size:16px">upload_file</span>
              上传文件
            </button>
            <button onclick="triggerIngest()" class="toolbar-btn">
              <span class="material-symbols-outlined" style="font-size:16px">folder_sync</span>
              扫描本地文件夹
            </button>
          </div>
        </div>
      </td></tr>`;
      renderLibraryPagination(0, 0, 0);
      renderLibraryContext();
      return;
    }
    tbody.innerHTML = visibleFiles.map(f => {
      const isStarred = Boolean(f.favorited) || favoritedIds.has(f.id);
      const favoriteLabel = isStarred ? `取消收藏 ${f.file_name}` : `收藏 ${f.file_name}`;
      const userTags = Array.isArray(f.user_tags) ? f.user_tags : [];
      const rowClass = f.id === activeLibraryFileId ? 'data-row-active' : 'hover:bg-surface-container/50';
      return `<tr data-file-id="${f.id}" onclick="setLibraryActiveFile(${f.id})" class="${rowClass} cursor-pointer transition-colors">
        <td class="py-3 pr-4"><input type="checkbox" data-id="${f.id}" ${f.status !== 'done' ? 'disabled' : ''} onchange="toggleFileSelect(this)" onclick="event.stopPropagation()" class="rounded"></td>
        <td class="py-3 pr-4 min-w-[220px]">
          <div class="text-sm font-semibold text-on-surface line-clamp-1">${escHtml(f.file_name)}</div>
          <div class="text-[11px] text-on-surface-variant/50 line-clamp-1">${escHtml(fileLocationLabel(f))}</div>
        </td>
        <td class="py-3 pr-4 text-xs text-on-surface-variant">${collectionBadge(f.collection)}</td>
        <td class="py-3 pr-4 text-xs text-on-surface-variant">${tagPills(userTags)}</td>
        <td class="py-3 pr-4 text-xs text-on-surface-variant">${fileTypeLabel(f.file_name, f.is_scanned)}</td>
        <td class="py-3 pr-4">${statusBadge(f.status)}</td>
        <td class="py-3 pr-4 text-sm text-on-surface-variant">${f.total_pages}</td>
        <td class="py-3 pr-4 text-sm text-on-surface-variant">${f.chunk_count}</td>
        <td class="py-3 pr-4 text-xs text-on-surface-variant/60">${(f.updated_at || '').slice(0,16) || '-'}</td>
        <td class="py-3">
          <div class="flex items-center justify-end gap-2">
          <button onclick="event.stopPropagation(); openFilePreview(${f.id})" class="text-on-surface-variant/40 hover:text-primary transition-colors" title="打开预览" aria-label="打开预览">
            <span class="material-symbols-outlined" style="font-size:18px">open_in_new</span>
          </button>
          <button onclick="event.stopPropagation(); toggleFavorite(${f.id}, this)" class="text-on-surface-variant/30 hover:text-primary transition-colors" title="${escHtml(favoriteLabel)}" aria-label="${escHtml(favoriteLabel)}">
            <span class="material-symbols-outlined ${isStarred ? 'icon-fill text-primary' : ''}" style="font-size:18px">star</span>
          </button>
          </div>
        </td>
      </tr>`;
    }).join('');
    document.getElementById('select-all').checked = false;
    renderLibraryPagination(files.length, start + 1, Math.min(start + libraryPageSize, files.length));
    renderLibraryContext();
  } catch (e) {
    if (requestId !== refreshFilesRequestId) return;
    document.getElementById('file-tbody').innerHTML =
      `<tr><td colspan="10" class="py-8 text-center text-error text-sm">加载失败：${e.message}</td></tr>`;
    renderLibraryContext({ error: e.message });
  } finally {
    if (showLoading && requestId === refreshFilesRequestId) setRefreshButtonLoading(false);
  }
}

function renderLibraryPagination(total, start, end) {
  const el = document.getElementById('library-pagination');
  if (!el) return;
  if (!total) {
    el.innerHTML = '<span>暂无文件</span><span>每页 14 条</span>';
    return;
  }
  const totalPages = Math.max(1, Math.ceil(total / libraryPageSize));
  const pages = Array.from({ length: Math.min(5, totalPages) }, (_, idx) => idx + 1);
  el.innerHTML = `
    <span>共 ${total} 个文件，显示 ${start}-${end}</span>
    <div class="flex items-center gap-1">
      <button onclick="setLibraryPage(${Math.max(1, libraryPage - 1)})" ${libraryPage <= 1 ? 'disabled' : ''} class="icon-button !w-8 !h-8 disabled:opacity-40" title="上一页" aria-label="上一页">
        <span class="material-symbols-outlined" style="font-size:14px">arrow_back</span>
      </button>
      ${pages.map(page => `<button onclick="setLibraryPage(${page})" class="w-8 h-8 rounded-lg text-xs font-bold ${page === libraryPage ? 'bg-primary-container text-primary' : 'hover:bg-surface-container'}">${page}</button>`).join('')}
      ${totalPages > 5 ? `<span class="px-2">… ${totalPages}</span>` : ''}
      <button onclick="setLibraryPage(${Math.min(totalPages, libraryPage + 1)})" ${libraryPage >= totalPages ? 'disabled' : ''} class="icon-button !w-8 !h-8 disabled:opacity-40" title="下一页" aria-label="下一页">
        <span class="material-symbols-outlined" style="font-size:14px">arrow_forward</span>
      </button>
      <span class="ml-3 text-[11px]">每页 ${libraryPageSize} 条</span>
    </div>`;
  renderLocalIcons(el);
}

function setLibraryPage(page) {
  libraryPage = page;
  refreshFiles({ preserveFilters: true });
}

function setLibraryActiveFile(fileId) {
  activeLibraryFileId = fileId;
  document.querySelectorAll('#file-tbody tr').forEach(row => {
    const active = row.dataset.fileId === String(fileId);
    row.classList.toggle('data-row-active', active);
  });
  renderLibraryContext();
}

function renderLibraryContext(state = {}) {
  const panel = document.getElementById('library-context-panel');
  if (!panel) return;
  if (state.error) {
    panel.innerHTML = `<section class="soft-panel p-4"><h2 class="panel-title">文件详情</h2><p class="mt-2 text-xs text-error">读取失败：${escHtml(state.error)}</p></section>`;
    return;
  }
  const file = libraryFiles.find(item => item.id === activeLibraryFileId);
  if (!file) {
    panel.innerHTML = `
      <section class="soft-panel p-4">
        <h2 class="panel-title">文件详情</h2>
        <p class="panel-muted mt-1">选择文件后显示来源、标签和入库状态。</p>
      </section>
      <section id="library-knowledge-overview" class="soft-panel p-4">
        <h2 class="panel-title">知识视图</h2>
        <div class="mt-3 text-xs text-on-surface-variant">正在整理主题、相似资料和知识卡片…</div>
      </section>`;
    loadKnowledgeOverview(null);
    return;
  }
  const tags = Array.isArray(file.user_tags) ? file.user_tags : [];
  const canRebuild = file.status === 'done';
  const statusText = fileStatusLabel(file.status);
  panel.innerHTML = `
    <section class="soft-panel p-4">
      <div class="flex items-start justify-between gap-3">
        <div class="min-w-0">
          <h2 class="panel-title line-clamp-2">${escHtml(file.file_name)}</h2>
          <p class="panel-muted mt-1">${escHtml(fileTypeLabel(file.file_name, file.is_scanned))} · ${escHtml(collectionLabel(file.collection))}</p>
        </div>
        <button onclick="toggleFavorite(${file.id}, this)" class="icon-button !w-8 !h-8" title="收藏文件" aria-label="收藏文件">
          <span class="material-symbols-outlined ${file.favorited ? 'icon-fill text-primary' : ''}" style="font-size:16px">star</span>
        </button>
      </div>
      <div class="mt-4 grid grid-cols-3 gap-2 text-xs">
        <div class="rounded-lg bg-surface-container-low px-3 py-2">
          <div class="panel-muted">页数</div>
          <div class="mt-1 font-bold text-on-surface">${file.total_pages ?? '-'}</div>
        </div>
        <div class="rounded-lg bg-surface-container-low px-3 py-2">
          <div class="panel-muted">片段</div>
          <div class="mt-1 font-bold text-on-surface">${file.chunk_count ?? '-'}</div>
        </div>
        <div class="rounded-lg bg-surface-container-low px-3 py-2">
          <div class="panel-muted">状态</div>
          <div class="mt-1 font-bold text-on-surface">${escHtml(statusText)}</div>
        </div>
      </div>
    </section>
    <section class="soft-panel p-2">
      <div class="grid grid-cols-4 gap-1 text-xs font-bold text-on-surface-variant">
        <button onclick="document.getElementById('library-detail-details')?.scrollIntoView({block:'nearest'})" class="rounded-lg bg-primary-container text-primary px-2 py-2">详情</button>
        <button onclick="document.getElementById('library-detail-preview')?.scrollIntoView({block:'nearest'})" class="rounded-lg hover:bg-surface-container px-2 py-2">预览</button>
        <button onclick="document.getElementById('library-detail-content')?.scrollIntoView({block:'nearest'})" class="rounded-lg hover:bg-surface-container px-2 py-2">内容</button>
        <button onclick="document.getElementById('library-detail-related')?.scrollIntoView({block:'nearest'})" class="rounded-lg hover:bg-surface-container px-2 py-2">关联</button>
      </div>
    </section>
    <section id="library-detail-details" class="soft-panel p-4">
      <h3 class="panel-title">所属集合</h3>
      <div class="mt-3 flex items-center justify-between gap-2 rounded-lg bg-surface-container-low px-3 py-3 text-xs">
        <span class="inline-flex items-center gap-2 font-semibold text-on-surface"><span class="material-symbols-outlined text-primary" style="font-size:15px">inventory_2</span>${escHtml(collectionLabel(file.collection))}</span>
        <button onclick="selectSingleAndOpenKnowledge(${file.id})" class="toolbar-btn !h-8" title="基于这个文件生成内容" aria-label="基于这个文件生成内容">使用</button>
      </div>
      <h3 class="panel-title mt-4">标签</h3>
      <div class="mt-3">${tagPills(tags)}</div>
    </section>
    <section id="library-detail-preview" class="soft-panel p-4">
      <h3 class="panel-title">来源预览</h3>
      <p class="mt-2 text-xs leading-relaxed text-on-surface-variant line-clamp-2">${escHtml(file.preview || fileLocationLabel(file) || '暂无预览')}</p>
      <div class="mt-3 flex flex-col gap-2">
        <button onclick="openSourcePreview(${file.id})" class="toolbar-btn toolbar-btn-primary" title="打开来源预览" aria-label="打开来源预览">
          <span class="material-symbols-outlined" style="font-size:15px">article</span>
          打开来源预览
        </button>
        <button id="library-source-review-btn" onclick="openSourceReview(${file.id})" class="toolbar-btn" title="查看引用片段" aria-label="查看引用片段">
          <span class="material-symbols-outlined" style="font-size:15px">link</span>
          查看引用片段
        </button>
        <button onclick="openFilePreview(${file.id})" class="toolbar-btn" title="打开原文预览" aria-label="打开原文预览">
          <span class="material-symbols-outlined" style="font-size:15px">open_in_new</span>
          打开原文
        </button>
      </div>
      <div id="library-source-review" class="mt-3 rounded-lg bg-surface-container-low px-3 py-3 text-xs text-on-surface-variant">${sourceReviewBodyMarkup(file.id)}</div>
    </section>
    <section id="library-detail-content" class="soft-panel p-4">
      <div class="flex items-center justify-between gap-3">
        <h3 class="panel-title">最近引用</h3>
        <button onclick="askWithLibraryFile(${file.id})" class="toolbar-btn" title="用这个文件提问" aria-label="用这个文件提问">
          <span class="material-symbols-outlined" style="font-size:15px">chat_bubble</span>
          提问
        </button>
      </div>
      <div id="library-recent-citations" class="mt-3 text-xs text-on-surface-variant">正在读取最近引用…</div>
    </section>
    <section id="library-detail-related" class="soft-panel p-4">
      <h3 class="panel-title">文件建议</h3>
      <div class="mt-3 flex flex-col gap-2">
        <button onclick="selectSingleAndOpenKnowledge(${file.id})" class="toolbar-btn" title="用这个文件生成知识产物" aria-label="用这个文件生成知识产物">
          <span class="material-symbols-outlined" style="font-size:15px">lightbulb</span>
          生成知识产物
        </button>
        <button onclick="selectSingleAndRebuild(${file.id})" ${canRebuild ? '' : 'disabled'} class="toolbar-btn disabled:opacity-45" title="重新整理这个文件" aria-label="重新整理这个文件">
          <span class="material-symbols-outlined" style="font-size:15px">sync</span>
          重新整理
        </button>
      </div>
    </section>
    <section id="library-knowledge-overview" class="soft-panel p-4">
      <h3 class="panel-title">知识视图</h3>
      <div class="mt-3 text-xs text-on-surface-variant">正在整理主题、相似资料和知识卡片…</div>
    </section>`;
  renderLocalIcons(panel);
  loadLibraryFileActivity(file);
  loadKnowledgeOverview(file.id);
}

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
    el.innerHTML = `<h3 class="panel-title">知识视图</h3><div class="mt-3 text-xs text-error">读取失败：${escHtml(e.message)}</div>`;
  }
}

function knowledgeOverviewMarkup(data) {
  const topics = data?.topics || [];
  const similar = data?.similar_documents || [];
  const cards = data?.knowledge_cards || [];
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
      error: e.message,
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
    if (!r.ok) throw new Error(await r.text());
    const panel = document.getElementById('library-source-review');
    if (panel) {
      panel.insertAdjacentHTML('afterbegin', '<div class="mb-2 rounded-lg bg-tertiary-container px-3 py-2 text-[11px] font-bold text-on-tertiary-container">已保存为笔记，并加入入库队列。</div>');
    }
    setScanButtonState('queued');
    startQueuePolling();
  } catch (e) {
    const panel = document.getElementById('library-source-review');
    if (panel) panel.insertAdjacentHTML('afterbegin', `<div class="mb-2 rounded-lg bg-error/10 px-3 py-2 text-[11px] font-bold text-error">保存失败：${escHtml(e.message)}</div>`);
  }
}

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
    el.innerHTML = `<span class="text-error">最近引用读取失败：${escHtml(e.message)}</span>`;
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
    if (!r.ok) throw new Error(await r.text());
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
    if (!r.ok) throw new Error(await r.text());
    await refreshFiles();
  } catch (e) {
    alert(`批量收藏失败: ${e.message}`);
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
    if (!r.ok) throw new Error(await r.text());
    document.getElementById('batch-collection-input').value = '';
    document.getElementById('batch-tags-input').value = '';
    await refreshFiles();
  } catch (e) {
    alert(`批量更新失败: ${e.message}`);
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
    if (!r.ok) throw new Error(await r.text());
    setScanButtonState('queued');
    pollQueueOnce();
    await refreshFiles();
  } catch (e) {
    alert(`整理失败: ${e.message}`);
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
    if (!r.ok) throw new Error(await r.text());
    const md = await r.text();
    const blob = new Blob([md], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'docflow-summary.md'; a.click();
    URL.revokeObjectURL(url);
  } catch (e) {
    alert(`摘要生成失败: ${e.message}`);
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<span class="material-symbols-outlined" style="font-size:15px">summarize</span><span class="ml-1">生成摘要</span>';
  }
}

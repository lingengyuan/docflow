async function refreshFiles(options = {}) {
  const showLoading = Boolean(options.showLoading);
  const requestId = ++refreshFilesRequestId;
  if (showLoading) setRefreshButtonLoading(true);
  try {
    const [files, meta, allFiles] = await Promise.all([
      fetch(`${API}/api/files${currentFileQuery()}`).then(r => {
        if (!r.ok) throw new Error('资料列表读取失败');
        return r.json();
      }),
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
    const message = userFacingErrorMessage(e.message, '资料列表暂时无法读取。');
    document.getElementById('file-tbody').innerHTML =
      `<tr><td colspan="10" class="py-8 text-center text-error text-sm">加载失败：${escHtml(message)}</td></tr>`;
    renderLibraryContext({ error: message });
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

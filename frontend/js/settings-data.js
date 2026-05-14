async function refreshSettings() {
  await Promise.all([
    checkHealth(),
    loadLLMOptions(),
    loadSettingsSources(),
    loadSettingsStorage(),
  ]);
}

async function loadSettingsSources() {
  const el = document.getElementById('settings-sources-list');
  if (!el) return;
  try {
    const sources = await fetch(`${API}/api/sources`).then(r => r.json());
    if (!sources.length) {
      el.innerHTML = '<div class="text-on-surface-variant/60">暂无监控目录</div>';
      return;
    }
    el.innerHTML = `
      <div class="overflow-x-auto rounded-lg bg-surface-container-low">
        <table class="w-full min-w-[520px] text-xs">
          <thead class="text-[11px] font-bold uppercase tracking-widest text-on-surface-variant/60">
            <tr>
              <th class="text-left px-3 py-2">目录</th>
              <th class="text-left px-3 py-2">范围</th>
              <th class="text-left px-3 py-2">状态</th>
              <th class="text-left px-3 py-2">包含子目录</th>
            </tr>
          </thead>
          <tbody class="divide-y divide-outline-variant/10">
            ${sources.map(source => `
              <tr>
                <td class="px-3 py-2 font-semibold text-on-surface">${escHtml(sourceDisplayName(source.path))}</td>
                <td class="px-3 py-2 text-on-surface-variant line-clamp-1">${source.recursive ? '包含子目录' : '仅当前文件夹'}</td>
                <td class="px-3 py-2 text-tertiary font-bold">监控中</td>
                <td class="px-3 py-2 text-on-surface-variant">${source.recursive ? '开启' : '关闭'} · ${(source.extensions || []).length || 0} 类文件</td>
              </tr>`).join('')}
          </tbody>
        </table>
      </div>`;
  } catch (e) {
    const message = userFacingErrorMessage(e.message, '监控目录暂时无法读取。');
    el.innerHTML = `<div class="text-error">监控目录读取失败：${escHtml(message)}</div>`;
  }
}

async function loadSettingsStorage() {
  const el = document.getElementById('settings-storage-list');
  if (!el) return;
  try {
    const [meta, allFiles, usage] = await Promise.all([
      fetch(`${API}/api/library/meta`).then(r => r.json()).catch(() => libraryMeta),
      fetch(`${API}/api/files`).then(r => r.json()).catch(() => []),
      fetch(`${API}/api/storage/usage`).then(r => {
        if (!r.ok) throw new Error('本地存储读取失败');
        return r.json();
      }),
    ]);
    libraryMeta = normalizeLibraryMeta(meta || libraryMeta, allFiles);
    storageUsage = usage || null;
    let browserBytes = 0;
    let browserDetail = '';
    if (navigator.storage?.estimate) {
      const estimate = await navigator.storage.estimate();
      browserBytes = Number(estimate.usage || 0);
      browserDetail = estimate.quota ? `上限 ${formatBytes(estimate.quota)}` : '浏览器本地缓存';
    }
    updateSidebarStorageSummary();
    const disk = storageUsage?.disk || {};
    const library = storageUsage?.library || {};
    const rawCategories = Array.isArray(storageUsage?.categories) ? storageUsage.categories : [];
    const categories = rawCategories.map(item => ({ ...item, bytes: Number(item.bytes || 0) }));
    if (browserBytes > 0) {
      const other = categories.find(item => item.id === 'other');
      if (other) other.bytes = Math.max(0, Number(other.bytes || 0) - browserBytes);
      categories.splice(Math.max(0, categories.findIndex(item => item.id === 'other')), 0, {
        id: 'browser',
        label: '浏览器缓存',
        bytes: browserBytes,
        detail: browserDetail,
      });
    }
    const totalBytes = Number(disk.total_bytes || 0);
    const usedBytes = Number(disk.used_bytes || categories.reduce((sum, item) => sum + item.bytes, 0));
    const freeBytes = Number(disk.free_bytes || Math.max(0, totalBytes - usedBytes));
    const usedPercent = storagePercent(usedBytes, totalBytes);
    const categoryTotal = Math.max(1, categories.reduce((sum, item) => sum + Number(item.bytes || 0), 0));
    const gradient = storageConicGradient(categories, categoryTotal);
    const categoryRows = categories.map(item => `
      <div class="flex items-start justify-between gap-3 rounded-lg bg-surface-container-low px-3 py-2">
        <div class="min-w-0">
          <div class="flex items-center gap-1.5 font-semibold text-on-surface">
            <span class="inline-block h-2 w-2 rounded-full" style="background:${storageCategoryColor(item.id)}"></span>
            <span>${escHtml(storageCategoryLabel(item))}</span>
          </div>
          <div class="mt-1 text-[11px] leading-relaxed text-on-surface-variant/65">${escHtml(item.detail || '')}</div>
        </div>
        <div class="whitespace-nowrap font-bold text-on-surface">${formatBytes(item.bytes || 0)}</div>
      </div>`).join('');
    el.innerHTML = `
      <div class="rounded-lg bg-surface-container-low px-3 py-3">
        <div class="flex items-start justify-between gap-3">
          <div>
            <div class="text-[11px] font-bold uppercase tracking-widest text-on-surface-variant/55">本地存储</div>
            <div class="mt-1 text-lg font-bold text-on-surface">${formatBytes(usedBytes)} / ${formatBytes(totalBytes)}</div>
          </div>
          <div class="rounded-full bg-primary/10 px-2.5 py-1 text-[11px] font-bold text-primary">${usedPercent}% 已用</div>
        </div>
        <div class="mt-3 h-2 overflow-hidden rounded-full bg-white">
          <div class="h-full rounded-full bg-primary transition-all" style="width:${usedPercent}%"></div>
        </div>
        <div class="mt-2 flex items-center justify-between gap-3 text-[11px] text-on-surface-variant/65">
          <span>可用 ${formatBytes(freeBytes)}</span>
          <span>${library.file_count ?? libraryMeta.total_files ?? 0} 个文件 · ${library.collection_count ?? (libraryMeta.collections || []).length} 个集合</span>
        </div>
      </div>
      <div class="rounded-lg bg-surface-container-low px-3 py-3">
        <div class="flex items-center gap-4">
          <div class="relative h-20 w-20 flex-shrink-0 rounded-full" style="background:${gradient};">
            <div class="absolute inset-4 rounded-full bg-surface-container-low"></div>
          </div>
          <div class="min-w-0 flex-1">
            <div class="font-semibold text-on-surface">占用来源</div>
            <div class="mt-1 text-[11px] leading-relaxed text-on-surface-variant/65">资料、模型和本机其他内容的合计。</div>
          </div>
        </div>
      </div>
      ${categoryRows || '<div class="rounded-lg bg-surface-container-low px-3 py-2">暂无存储数据</div>'}`;
  } catch (e) {
    const message = userFacingErrorMessage(e.message, '存储统计暂时无法读取。');
    el.innerHTML = `<div class="text-error">存储统计读取失败：${escHtml(message)}</div>`;
  }
}

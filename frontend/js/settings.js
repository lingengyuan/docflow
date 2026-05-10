// ── Health check ──
async function checkHealth() {
  try {
    const r = await fetch(`${API}/api/health`);
    healthSnapshot = await r.json();
    const dot = document.getElementById('status-dot');
    const ok = r.ok && healthSnapshot.status !== 'unavailable';
    const degraded = healthSnapshot.status === 'degraded';
    const cls = ok ? (degraded ? 'bg-primary' : 'bg-tertiary') : 'bg-error';
    dot.className = `w-2 h-2 rounded-full flex-shrink-0 ${cls}`;
    document.getElementById('top-status-dot').className = `w-2 h-2 rounded-full ${cls}`;
    document.getElementById('health-icon').className = `w-2 h-2 rounded-full ${cls}`;
    const label = healthStatusLabel(healthSnapshot.status);
    document.getElementById('health-label').textContent = label;
    document.getElementById('top-status-label').textContent = label;
    document.getElementById('sidebar-status-label').textContent = label;
    document.getElementById('chat-context-health').textContent = label;
    const chatStatus = document.getElementById('chat-status-label');
    if (chatStatus) chatStatus.textContent = label;
    renderHealthPanel();
    renderSettingsInsights();
  } catch {
    document.getElementById('status-dot').className = 'w-2 h-2 rounded-full flex-shrink-0 bg-error';
    document.getElementById('top-status-dot').className = 'w-2 h-2 rounded-full bg-error';
    document.getElementById('health-icon').className = 'w-2 h-2 rounded-full bg-error';
    document.getElementById('health-label').textContent = '不可用';
    document.getElementById('top-status-label').textContent = '不可用';
    document.getElementById('sidebar-status-label').textContent = '不可用';
    document.getElementById('chat-context-health').textContent = '不可用';
    const chatStatus = document.getElementById('chat-status-label');
    if (chatStatus) chatStatus.textContent = '不可用';
    healthSnapshot = null;
    renderHealthPanel('无法读取状态');
    renderSettingsInsights('本地状态暂时无法读取。');
  }
}
checkHealth();
setInterval(checkHealth, 15000);

function toggleHealthPanel(e) {
  e.stopPropagation();
  checkHealth();
  document.getElementById('health-panel')?.classList.remove('hidden');
}

function renderHealthPanel(errorText = '') {
  const el = document.getElementById('health-details');
  if (!el) return;
  if (errorText) {
    el.innerHTML = `<div class="text-error">${escHtml(errorText)}</div>`;
    return;
  }
  if (!healthSnapshot) {
    el.innerHTML = '<div>正在检查…</div>';
    return;
  }
  const groups = healthSnapshot.groups || null;
  if (!groups) {
    el.innerHTML = '<div>暂无状态</div>';
    return;
  }
  el.innerHTML = ['core', 'runtime', 'optional'].map(key => renderHealthGroup(groups[key])).join('');
}

function healthStatusLabel(status) {
  if (status === 'ok') return '正常';
  if (status === 'degraded') return '核心可用';
  if (status === 'unavailable') return '不可用';
  return status || '状态';
}

function renderHealthGroup(group) {
  if (!group?.items?.length) return '';
  const rows = group.items.map(item => {
    const color = healthItemColor(item.status);
    const label = healthItemLabel(item.status);
    return `<div class="min-h-[54px] rounded-lg bg-surface-container-low px-3 py-2">
      <div class="flex items-center justify-between gap-3">
        <span class="inline-flex items-center gap-1.5 text-[11px] font-bold whitespace-nowrap">
          <span class="w-1.5 h-1.5 rounded-full ${color}"></span>${label}
        </span>
        <span class="text-[10px] text-on-surface-variant/55">${escHtml(group.label)}</span>
      </div>
      <div class="mt-1 min-w-0">
        <div class="font-semibold text-on-surface line-clamp-1">${escHtml(item.label || item.key)}</div>
        ${item.detail ? `<div class="text-[11px] text-on-surface-variant/70 line-clamp-1">${escHtml(settingsSafeDetail(item.detail))}</div>` : ''}
      </div>
    </div>`;
  }).join('');
  return rows;
}

function renderSettingsInsights(errorText = '') {
  const el = document.getElementById('settings-insights-list');
  if (!el) return;
  if (errorText) {
    el.innerHTML = `<div class="text-error">${escHtml(errorText)}</div>`;
    return;
  }
  if (!healthSnapshot) {
    el.innerHTML = '<div class="rounded-lg bg-surface-container-low px-3 py-2 text-on-surface-variant/60">正在读取本地状态…</div>';
    return;
  }
  const groups = healthSnapshot.groups || {};
  const optionalItems = groups.optional?.items || [];
  const unavailableOptional = optionalItems.filter(item => item.status !== 'ok');
  const insights = [];
  if (healthSnapshot.status === 'ok') {
    insights.push({ title: '核心功能正常', detail: '问答、入库和本地资料可以正常使用。' });
  } else if (healthSnapshot.status === 'degraded') {
    insights.push({ title: '核心功能可用', detail: '主要功能可用，部分增强能力暂未就绪。' });
  } else {
    insights.push({ title: '本地服务暂不可用', detail: '稍后刷新状态，或确认本地服务已经打开。' });
  }
  if (unavailableOptional.length) {
    insights.push({
      title: '增强能力未完全启用',
      detail: `${unavailableOptional.length} 项增强能力暂未就绪，不影响基础文件问答。`,
    });
  } else {
    insights.push({ title: '增强能力已就绪', detail: '图片理解、模型状态和本地能力检查都已完成。' });
  }
  el.innerHTML = insights.map(item => `
    <div class="rounded-lg bg-surface-container-low px-3 py-2">
      <div class="font-semibold text-on-surface">${escHtml(item.title)}</div>
      <div class="text-[11px] text-on-surface-variant/60 mt-0.5">${escHtml(item.detail)}</div>
    </div>`).join('');
}

function settingsSafeDetail(detail) {
  const text = String(detail || '');
  if (!text) return '';
  const developerTokens = [
    ['python main', 'py'].join('.'),
    ['--dry', 'run'].join('-'),
    ['install', 'local'].join('-'),
    ['restore', 'drill'].join('-'),
    ['repair', 'ids'].join('-'),
    ['browser', 'acceptance'].join('-'),
    'doctor',
    'docker ',
    ['ollama', 'pull'].join(' '),
    'Run:',
  ];
  if (developerTokens.some(token => text.includes(token))) {
    return '本地状态已记录；需要时可以刷新状态或查看帮助说明。';
  }
  return text;
}

function healthItemColor(status) {
  if (status === 'ok') return 'bg-tertiary';
  if (status === 'off') return 'bg-on-surface-variant/30';
  if (status === 'optional_unavailable') return 'bg-primary';
  if (status === 'degraded') return 'bg-primary';
  return 'bg-error';
}

function healthItemLabel(status) {
  if (status === 'ok') return '可用';
  if (status === 'off') return '未启用';
  if (status === 'optional_unavailable') return '未安装';
  if (status === 'degraded') return '降级';
  return '不可用';
}

// ── LLM selector ──
async function loadLLMOptions() {
  try {
    const d = await fetch(`${API}/api/llm`).then(r => r.json());
    const models = d.models || (d.options || []).map(m => ({ model: m, available: true, cached: true, current: m === d.current }));
    llmOptions = models.map(m => m.model);
    document.getElementById('llm-current').textContent = d.current || 'LLM';
    const chatModel = document.getElementById('chat-context-model');
    if (chatModel) chatModel.textContent = d.current || 'LLM';
    const knowledgeModel = document.getElementById('knowledge-model-select');
    if (knowledgeModel) {
      knowledgeModel.innerHTML = models
        .filter(item => item.available)
        .map(item => `<option value="${escHtml(item.model)}" ${item.current ? 'selected' : ''}>${escHtml(item.model)}</option>`)
        .join('') || `<option>${escHtml(d.current || '本地模型')}</option>`;
    }
    renderLLMStatus(d.switch);
    renderSettingsModelList(d);

    const dropdown = document.getElementById('llm-dropdown');
    dropdown.innerHTML = models.map(item => `
      <button data-model="${escHtml(item.model)}" onclick="switchLLM(this.dataset.model)" ${item.available ? '' : 'disabled'}
        class="w-full text-left px-4 py-2.5 text-xs font-medium transition-all ${item.current ? 'text-primary font-bold' : 'text-on-surface'} ${item.available ? 'hover:bg-surface-container' : 'opacity-45 cursor-not-allowed'}">
        <span class="block">${escHtml(item.model)}</span>
        <span class="block text-[10px] font-normal text-on-surface-variant/60">${item.current ? '当前使用' : (item.available ? '本地可用' : '未缓存')}</span>
      </button>`).join('');
  } catch {
    renderLLMStatus({ state: 'error', message: '模型状态读取失败' });
  }
}
initLocalIcons();
loadLLMOptions();
loadConversations();
loadQueryScopeOptions();
loadLatestAnswerPreview();
syncKnowledgeOutputCards();

document.getElementById('llm-btn').addEventListener('click', (e) => {
  e.stopPropagation();
  document.getElementById('llm-dropdown').classList.toggle('hidden');
});
document.addEventListener('click', () => {
  document.getElementById('llm-dropdown').classList.add('hidden');
  document.getElementById('conversation-menu').classList.add('hidden');
});

async function switchLLM(model) {
  const current = document.getElementById('llm-current').textContent;
  if (model === current) {
    document.getElementById('llm-dropdown').classList.add('hidden');
    renderLLMStatus({ state: 'idle', message: '当前模型' });
    return;
  }
  try {
    document.getElementById('llm-dropdown').classList.add('hidden');
    document.getElementById('llm-btn').disabled = true;
    document.getElementById('llm-current').textContent = '切换中…';
    renderLLMStatus({ state: 'switching', model });
    const r = await fetch(`${API}/api/llm`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ model }),
    });
    const d = await r.json();
    if (!r.ok || !d.ok) {
      throw new Error(d.detail || '模型切换失败');
    }
    if (d.ok) {
      document.getElementById('llm-current').textContent = model;
      renderLLMStatus({ state: 'idle', message: '本地模型' });
      loadLLMOptions();
    }
  } catch (e) {
    renderLLMStatus({ state: 'error', message: e.message });
    loadLLMOptions();
  } finally {
    document.getElementById('llm-btn').disabled = false;
  }
}

function handleKnowledgeModelSelect() {
  const model = document.getElementById('knowledge-model-select')?.value || '';
  if (model) switchLLM(model);
}

function renderLLMStatus(state) {
  const el = document.getElementById('llm-status');
  if (!el) return;
  el.classList.remove('text-error', 'text-primary', 'text-on-surface-variant');
  if (state?.state === 'switching') {
    el.textContent = '切换中';
    el.classList.add('text-primary');
  } else if (state?.state === 'error') {
    el.textContent = state.message || '切换失败';
    el.classList.add('text-error');
  } else {
    el.textContent = '本地模型';
    el.classList.add('text-on-surface-variant');
  }
}

function renderSettingsModelList(data) {
  const el = document.getElementById('settings-model-list');
  if (!el) return;
  const models = data?.models || [];
  if (!models.length) {
    el.innerHTML = '<div class="text-on-surface-variant/60">暂无模型信息</div>';
    return;
  }
  el.innerHTML = `
    <div class="overflow-x-auto rounded-lg bg-surface-container-low">
      <table class="w-full min-w-[560px] text-xs">
        <thead class="text-[11px] font-bold uppercase tracking-widest text-on-surface-variant/60">
          <tr>
            <th class="text-left px-3 py-2">模型类型</th>
            <th class="text-left px-3 py-2">状态</th>
            <th class="text-left px-3 py-2">当前模型</th>
            <th class="text-left px-3 py-2">缓存</th>
            <th class="text-left px-3 py-2">操作</th>
          </tr>
        </thead>
        <tbody class="divide-y divide-outline-variant/10">
          ${models.map(item => `
            <tr>
              <td class="px-3 py-2 font-semibold text-on-surface">${settingsModelTypeLabel(item.model)}</td>
              <td class="px-3 py-2">
                <span class="inline-flex items-center gap-1.5 font-bold ${item.available ? 'text-tertiary' : 'text-primary'}">
                  <span class="w-1.5 h-1.5 rounded-full ${item.available ? 'bg-tertiary' : 'bg-primary'}"></span>${item.current ? '当前' : (item.available ? '可用' : '未缓存')}
                </span>
              </td>
              <td class="px-3 py-2 text-on-surface-variant">${escHtml(item.model)}</td>
              <td class="px-3 py-2 text-on-surface-variant">${escHtml(item.size || item.detail || '-')}</td>
              <td class="px-3 py-2">
                <button onclick="switchLLM(decodeURIComponent('${encodeURIComponent(item.model)}'))" ${item.available && !item.current ? '' : 'disabled'} class="text-primary font-bold disabled:text-on-surface-variant/35">${item.current ? '已启用' : '切换'}</button>
              </td>
            </tr>`).join('')}
        </tbody>
      </table>
    </div>`;
}

function settingsModelTypeLabel(model) {
  const value = String(model || '').toLowerCase();
  if (value.includes('embed') || value.includes('nomic')) return 'Embedding（嵌入）';
  if (value.includes('rerank') || value.includes('bge')) return 'Reranker（重排）';
  if (value.includes('vl') || value.includes('vision') || value.includes('ocr')) return '图片理解';
  return 'LLM（大语言模型）';
}

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
              <th class="text-left px-3 py-2">路径</th>
              <th class="text-left px-3 py-2">状态</th>
              <th class="text-left px-3 py-2">包含子目录</th>
            </tr>
          </thead>
          <tbody class="divide-y divide-outline-variant/10">
            ${sources.map(source => `
              <tr>
                <td class="px-3 py-2 font-semibold text-on-surface">${escHtml(sourceDisplayName(source.path))}</td>
                <td class="px-3 py-2 text-on-surface-variant line-clamp-1">${escHtml(source.path || '')}</td>
                <td class="px-3 py-2 text-tertiary font-bold">监控中</td>
                <td class="px-3 py-2 text-on-surface-variant">${source.recursive ? '开启' : '关闭'} · ${(source.extensions || []).length || 0} 类文件</td>
              </tr>`).join('')}
          </tbody>
        </table>
      </div>`;
  } catch (e) {
    el.innerHTML = `<div class="text-error">监控目录读取失败：${escHtml(e.message)}</div>`;
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
    el.innerHTML = `<div class="text-error">存储统计读取失败：${escHtml(e.message)}</div>`;
  }
}

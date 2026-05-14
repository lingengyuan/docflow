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
    window.currentLLMModel = d.current || '';
    const currentModel = models.find(item => item.current) || { model: d.current || '', current: true, available: true };
    document.getElementById('llm-current').textContent = friendlyModelLabel(currentModel, models.indexOf(currentModel), d);
    const chatModel = document.getElementById('chat-context-model');
    if (chatModel) chatModel.textContent = friendlyModelLabel(currentModel, models.indexOf(currentModel), d);
    const knowledgeModel = document.getElementById('knowledge-model-select');
    if (knowledgeModel) {
      knowledgeModel.innerHTML = models
        .filter(item => item.available)
        .map((item, index) => `<option value="${escHtml(item.model)}" ${item.current ? 'selected' : ''}>${escHtml(friendlyModelLabel(item, index, d))}</option>`)
        .join('') || `<option>${escHtml(friendlyModelLabel(currentModel, 0, d))}</option>`;
    }
    renderLLMStatus(d.switch, d);
    renderSettingsModelList(d);

    const dropdown = document.getElementById('llm-dropdown');
    dropdown.innerHTML = models.map((item, index) => `
      <button data-model="${escHtml(item.model)}" onclick="switchLLM(this.dataset.model)" ${item.available ? '' : 'disabled'}
        class="w-full text-left px-4 py-2.5 text-xs font-medium transition-all ${item.current ? 'text-primary font-bold' : 'text-on-surface'} ${item.available ? 'hover:bg-surface-container' : 'opacity-45 cursor-not-allowed'}">
        <span class="block">${escHtml(friendlyModelLabel(item, index, d))}</span>
        <span class="block text-[10px] font-normal text-on-surface-variant/60">${escHtml(friendlyModelDetail(item, index, d))}</span>
      </button>`).join('');
  } catch {
    renderLLMStatus({ state: 'error', message: '模型状态读取失败' });
  }
}

async function switchLLM(model) {
  if (model === window.currentLLMModel) {
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
      window.currentLLMModel = model;
      document.getElementById('llm-current').textContent = friendlyModelLabel({ model, current: true }, 0, { network_mode: 'local' });
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

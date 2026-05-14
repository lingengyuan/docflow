function renderLLMStatus(state, data = null) {
  const el = document.getElementById('llm-status');
  if (!el) return;
  el.classList.remove('text-error', 'text-primary', 'text-on-surface-variant');
  if (state?.state === 'switching') {
    el.textContent = '切换中';
    el.classList.add('text-primary');
  } else if (state?.state === 'error') {
    el.textContent = state.message || '切换失败';
    el.classList.add('text-error');
  } else if (data?.network_mode === 'cloud') {
    el.textContent = '云端回答已启用';
    el.classList.add('text-primary');
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
  const privacyNotice = data?.network_mode === 'cloud'
    ? `<div class="mb-2 rounded-lg bg-primary/10 px-3 py-2 text-xs text-on-surface">
        <div class="font-semibold">云端回答已启用</div>
        <div class="mt-0.5 text-on-surface-variant/70">${escHtml(data.privacy_notice || '提问内容会发送到你配置的外部模型服务。')}</div>
      </div>`
    : '';
  el.innerHTML = `
    ${privacyNotice}
    <div class="overflow-x-auto rounded-lg bg-surface-container-low">
      <table class="w-full min-w-[560px] text-xs">
        <thead class="text-[11px] font-bold uppercase tracking-widest text-on-surface-variant/60">
          <tr>
            <th class="text-left px-3 py-2">能力</th>
            <th class="text-left px-3 py-2">状态</th>
            <th class="text-left px-3 py-2">说明</th>
            <th class="text-left px-3 py-2">操作</th>
          </tr>
        </thead>
        <tbody class="divide-y divide-outline-variant/10">
          ${models.map((item, index) => `
            <tr>
              <td class="px-3 py-2 font-semibold text-on-surface">${escHtml(friendlyModelLabel(item, index, data))}</td>
              <td class="px-3 py-2">
                <span class="inline-flex items-center gap-1.5 font-bold ${item.available ? 'text-tertiary' : 'text-primary'}">
                  <span class="w-1.5 h-1.5 rounded-full ${item.available ? 'bg-tertiary' : 'bg-primary'}"></span>${item.current ? '当前' : (item.available ? '可用' : '未缓存')}
                </span>
              </td>
              <td class="px-3 py-2 text-on-surface-variant">${escHtml(friendlyModelDetail(item, index, data))}</td>
              <td class="px-3 py-2">
                <button onclick="switchLLM(decodeURIComponent('${encodeURIComponent(item.model)}'))" ${item.available && !item.current ? '' : 'disabled'} class="text-primary font-bold disabled:text-on-surface-variant/35">${item.current ? '已启用' : '切换'}</button>
              </td>
            </tr>`).join('')}
        </tbody>
      </table>
    </div>`;
}

function friendlyModelLabel(item, index = 0, data = null) {
  if (data?.network_mode === 'cloud') return item?.current ? '云端回答模型' : '备用云端模型';
  if (item?.current) return '本地回答模型';
  if (index === 1) return '增强回答模型';
  return '备用回答模型';
}

function friendlyModelDetail(item, index = 0, data = null) {
  if (data?.network_mode === 'cloud') {
    return item?.current ? '回答会使用你配置的外部模型服务。' : '可切换的外部模型服务。';
  }
  if (!item?.available) return '本机还没有准备好这个模型。';
  if (item?.current) return '正在用于问答和知识产物生成。';
  if (index === 1) return '适合更复杂的问题，可按需切换。';
  return '已在本机准备好，可按需切换。';
}

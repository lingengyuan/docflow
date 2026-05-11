// ── Shared UI helpers ──
function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}

function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
}

function activateByKeyboard(event, action) {
  if (event.key !== 'Enter' && event.key !== ' ') return;
  event.preventDefault();
  action();
}

function handleUploadZoneKey(event) {
  activateByKeyboard(event, () => document.getElementById('file-input').click());
}

function handleConversationKey(event, id) {
  activateByKeyboard(event, () => switchConversation(id));
}

function handleSourceKey(event, filePath, pageNum, chunkId = '', charStart = 0, charEnd = 0) {
  activateByKeyboard(event, () => openSourceByPath(filePath, pageNum, chunkId, charStart, charEnd));
}

function handleHistoryKey(event, idx) {
  activateByKeyboard(event, () => replayHistory(idx));
}

function escHtml(s) {
  if (!s) return '';
  return String(s)
    .replace(/&/g,'&amp;')
    .replace(/</g,'&lt;')
    .replace(/>/g,'&gt;')
    .replace(/"/g,'&quot;')
    .replace(/'/g,'&#39;');
}

function sourceDisplayName(filePath) {
  const parts = String(filePath || '').split(/[\\/]/).filter(Boolean);
  return parts.at(-2) || parts.at(-1) || '本地资料';
}

function collectionLabel(collection) {
  const value = String(collection || 'Inbox');
  const labels = {
    Inbox: '收件箱',
    Notes: '笔记',
    'Web Imports': '网页导入',
  };
  labels[['Saved', 'Answers'].join(' ')] = '已保存回答';
  labels[['Saved', 'Sources'].join(' ')] = '来源片段';
  labels[knowledgeOutputCollectionName()] = '知识产物';
  return labels[value] || value;
}

function knowledgeOutputCollectionName() {
  return ['Knowledge', 'Outputs'].join(' ');
}

function fileLocationLabel(file) {
  const folder = sourceDisplayName(file?.file_path);
  const collection = collectionLabel(file?.collection);
  return `${folder} · ${collection}`;
}

function formatBytes(value) {
  const bytes = Number(value || 0);
  if (bytes < 1024) return `${bytes} B`;
  const units = ['KB', 'MB', 'GB', 'TB'];
  let size = bytes / 1024;
  let unit = units[0];
  for (let i = 1; i < units.length && size >= 1024; i += 1) {
    size /= 1024;
    unit = units[i];
  }
  return `${size.toFixed(size >= 10 ? 0 : 1)} ${unit}`;
}

function storagePercent(usedBytes, totalBytes) {
  const total = Number(totalBytes || 0);
  if (total <= 0) return 0;
  return Math.max(0, Math.min(100, Math.round((Number(usedBytes || 0) / total) * 100)));
}

function storageCategoryColor(id) {
  return {
    library: '#0e8f8d',
    models: '#6ea4d8',
    app_data: '#54b98d',
    browser: '#e6b450',
    other: '#9aa7b6',
  }[id] || '#9aa7b6';
}

function storageCategoryLabel(item) {
  return item?.label || {
    library: '资料库文件',
    models: '模型缓存',
    app_data: '应用数据',
    browser: '浏览器缓存',
    other: '其他本地占用',
  }[item?.id] || '本地占用';
}

function storageConicGradient(categories, totalBytes) {
  let current = 0;
  const segments = categories
    .filter(item => Number(item.bytes || 0) > 0)
    .map(item => {
      const start = current;
      current += (Number(item.bytes || 0) / totalBytes) * 360;
      return `${storageCategoryColor(item.id)} ${start.toFixed(1)}deg ${current.toFixed(1)}deg`;
    });
  if (!segments.length) return 'conic-gradient(#d8e0e5 0deg 360deg)';
  if (current < 360) segments.push(`#d8e0e5 ${current.toFixed(1)}deg 360deg`);
  return `conic-gradient(${segments.join(', ')})`;
}

function updateSidebarStorageSummary() {
  const count = document.getElementById('sidebar-storage-count');
  const meter = document.getElementById('sidebar-storage-meter');
  if (!count) return;
  const disk = storageUsage?.disk || null;
  if (disk?.total_bytes) {
    const percent = storagePercent(disk.used_bytes, disk.total_bytes);
    count.textContent = `${formatBytes(disk.used_bytes)} / ${formatBytes(disk.total_bytes)}`;
    if (meter) meter.style.width = `${percent}%`;
    return;
  }
  const total = libraryMeta.total_files || 0;
  const collections = (libraryMeta.collections || []).length;
  count.textContent = `${total} 个文件 · ${collections} 个集合`;
  if (meter) meter.style.width = '0%';
}

function deriveLibraryMetaCounts(files = []) {
  const rows = Array.isArray(files) ? files : [];
  const types = { pdf: 0, markdown: 0, image: 0, code: 0 };
  rows.forEach(file => {
    const name = file?.file_name || '';
    const ext = name.split('.').pop().toLowerCase();
    if (ext === 'pdf') types.pdf += 1;
    else if (ext === 'md' || ext === 'markdown') types.markdown += 1;
    else if (['jpg','jpeg','png','webp','heic','heif'].includes(ext)) types.image += 1;
    else if (['py','rs','ts','tsx','js','jsx','css','sh'].includes(ext)) types.code += 1;
  });
  return {
    total_files: rows.length,
    favorites: rows.filter(file => file?.favorited).length,
    recent: Math.min(rows.length, 30),
    types,
  };
}

function normalizeLibraryMeta(meta, files = []) {
  const derived = deriveLibraryMetaCounts(files);
  const incomingTypes = meta?.types || {};
  const hasUsefulTypes = ['pdf', 'markdown', 'image', 'code'].some(key => Number(incomingTypes[key] || 0) > 0);
  return {
    ...(meta || {}),
    total_files: Number.isFinite(meta?.total_files) ? meta.total_files : derived.total_files,
    favorites: Number.isFinite(meta?.favorites) ? meta.favorites : derived.favorites,
    recent: Number.isFinite(meta?.recent) ? meta.recent : derived.recent,
    types: hasUsefulTypes ? incomingTypes : derived.types,
  };
}

function renderInlineMarkdown(line) {
  const codeSpans = [];
  let rendered = line.replace(/`([^`]+)`/g, (_, code) => {
    const token = `@@CODE_SPAN_${codeSpans.length}@@`;
    codeSpans.push(`<code>${code}</code>`);
    return token;
  });
  rendered = rendered
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer" class="text-primary underline underline-offset-2">$1</a>');
  return rendered.replace(/@@CODE_SPAN_(\d+)@@/g, (_, index) => codeSpans[Number(index)] || '');
}

function splitMarkdownTableRow(line) {
  return line.replace(/^\s*\|?|\|?\s*$/g, '').split('|').map(cell => cell.trim());
}

function isMarkdownTableSeparator(line) {
  return /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(line);
}

function renderMarkdown(text) {
  const lines = escHtml(text || '').split('\n');
  const html = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    const next = lines[i + 1] || '';
    if (/^\s*```/.test(line)) {
      const codeLines = [];
      i += 1;
      while (i < lines.length && !/^\s*```/.test(lines[i])) {
        codeLines.push(lines[i]);
        i += 1;
      }
      if (i < lines.length) i += 1;
      html.push(`<pre class="my-3 overflow-x-auto rounded-lg bg-surface-container-low px-3 py-2 text-xs leading-relaxed"><code>${codeLines.join('\n')}</code></pre>`);
      continue;
    }
    if (line.includes('|') && isMarkdownTableSeparator(next)) {
      const headers = splitMarkdownTableRow(line);
      i += 2;
      const rows = [];
      while (i < lines.length && lines[i].includes('|') && lines[i].trim()) {
        rows.push(splitMarkdownTableRow(lines[i]));
        i += 1;
      }
      html.push(`<table class="reference-table my-3"><thead><tr>${headers.map(cell => `<th>${renderInlineMarkdown(cell)}</th>`).join('')}</tr></thead><tbody>${rows.map(row => `<tr>${row.map(cell => `<td>${renderInlineMarkdown(cell)}</td>`).join('')}</tr>`).join('')}</tbody></table>`);
      continue;
    }
    const heading = line.match(/^\s*(#{1,3})\s+(.+)$/);
    if (heading) {
      const level = heading[1].length + 2;
      html.push(`<h${level} class="mt-4 mb-2 font-bold text-on-surface">${renderInlineMarkdown(heading[2])}</h${level}>`);
      i += 1;
      continue;
    }
    if (/^\s*&gt;\s?/.test(line)) {
      const items = [];
      while (i < lines.length && /^\s*&gt;\s?/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*&gt;\s?/, ''));
        i += 1;
      }
      html.push(`<blockquote class="my-3 border-l-2 border-primary/40 pl-3 text-on-surface-variant">${items.map(item => renderInlineMarkdown(item)).join('<br>')}</blockquote>`);
      continue;
    }
    if (/^\s*\d+\.\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^\s*\d+\.\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*\d+\.\s+/, ''));
        i += 1;
      }
      html.push(`<ol>${items.map(item => `<li>${renderInlineMarkdown(item)}</li>`).join('')}</ol>`);
      continue;
    }
    if (/^\s*[-*]\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^\s*[-*]\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*[-*]\s+/, ''));
        i += 1;
      }
      html.push(`<ul>${items.map(item => `<li>${renderInlineMarkdown(item)}</li>`).join('')}</ul>`);
      continue;
    }
    if (!line.trim()) {
      html.push('<br>');
    } else {
      html.push(`<p>${renderInlineMarkdown(line)}</p>`);
    }
    i += 1;
  }
  return html.join('');
}

function userFacingErrorMessage(raw, fallback = '处理失败，请稍后再试。') {
  let text = String(raw || '').trim();
  if (!text) return fallback;
  try {
    const parsed = JSON.parse(text);
    if (typeof parsed === 'string') text = parsed;
    else text = parsed?.detail || parsed?.message || fallback;
  } catch {}
  const lower = text.toLowerCase();
  if (text.includes('模型任务超时') || lower.includes('timeout')) {
    return '这次回答耗时太久，已停止。可以缩小范围后重试。';
  }
  if (lower.includes('failed to fetch') || lower.includes('networkerror')) {
    return '暂时连接不上本地服务，请确认 DocFlow 正在运行。';
  }
  if (lower.includes('query engine not ready') || lower.includes('store not ready') || lower.includes('not ready')) {
    return '本地服务还在准备，请稍后再试。';
  }
  if (lower.includes('conversation not found')) {
    return '这个对话已经不可用，请新建对话后再试。';
  }
  if (lower.includes('unknown model') || lower.includes('not cached')) {
    return '当前模型暂不可用，请在设置里选择一个可用模型。';
  }
  if (text.length > 120 || /traceback|exception|stack|json|detail|http|localhost|127\.0\.0\.1/i.test(text)) {
    return fallback;
  }
  return text;
}

async function responseUserMessage(response, fallback) {
  return userFacingErrorMessage(await response.text(), fallback);
}

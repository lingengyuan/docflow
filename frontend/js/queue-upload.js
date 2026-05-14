// ── Queue polling ──
function startQueuePolling() {
  if (queuePollTimer2) return;
  queuePollTimer2 = setInterval(pollQueueOnce, 2000);
}

function stopQueuePolling() {
  if (queuePollTimer2) { clearInterval(queuePollTimer2); queuePollTimer2 = null; }
  document.getElementById('queue-banner')?.classList.add('hidden');
  const queueContext = document.getElementById('chat-context-queue');
  if (queueContext) queueContext.textContent = '暂无后台任务。';
}

function isQueueActive(q) {
  return Boolean(q?.queue_size > 0 || q?.processing || q?.paused);
}

function queueFilesRefreshKey(q) {
  return [
    isQueueActive(q) ? 'active' : 'idle',
    q?.processing || '',
    Number(q?.queue_size || 0),
    q?.paused ? 'paused' : '',
    q?.progress?.stage || '',
    Number(q?.progress?.processed_chunks || 0),
    Number(q?.progress?.total_chunks || 0),
  ].join('|');
}

function maybeRefreshFilesForQueue(q) {
  const key = queueFilesRefreshKey(q);
  if (key === lastQueueFilesRefreshKey) return;
  lastQueueFilesRefreshKey = key;
  refreshFiles({ preserveFilters: true });
}

async function pollQueueOnce() {
  try {
    const q = await fetch(`${API}/api/queue`).then(r => r.json());
    const banner = document.getElementById('queue-banner');
    const textEl = document.getElementById('queue-text');
    if (q.queue_size > 0 || q.processing || q.paused) {
      banner.classList.remove('hidden');
      banner.classList.add('flex');
      const parts = [];
      const stageLabels = {
        queued: '排队中',
        preparing: '解析/切块',
        embedding: '建立索引',
        storing: '保存片段',
        processing: '处理中',
        paused: '前台问答中，后台让路',
      };
      if (q.paused) parts.push('前台问答中，后台整理暂停');
      if (q.processing) parts.push(`处理：${q.processing}`);
      if (q.progress?.stage) parts.push(`阶段：${stageLabels[q.progress.stage] || q.progress.stage}`);
      if (q.progress?.total_chunks > 0) {
        parts.push(`片段 ${q.progress.processed_chunks}/${q.progress.total_chunks}`);
      }
      if (q.progress?.batch_files?.length > 1) {
        parts.push(`微批 ${q.progress.batch_files.length} 个文件`);
      }
      if (q.queue_size > 0) parts.push(`队列 ${q.queue_size} 个`);
      textEl.textContent = parts.join(' · ');
      const queueContext = document.getElementById('chat-context-queue');
      if (queueContext) queueContext.textContent = parts.join(' · ');
      maybeRefreshFilesForQueue(q);
    } else {
      maybeRefreshFilesForQueue(q);
      stopQueuePolling();
      const queueContext = document.getElementById('chat-context-queue');
      if (queueContext) queueContext.textContent = '暂无后台任务。';
    }
  } catch {}
}

async function triggerIngest() {
  setScanButtonState('loading');
  try {
    const r = await fetch(`${API}/api/ingest`, { method: 'POST' });
    if (!r.ok) throw new Error(await responseUserMessage(r, '扫描文件夹失败，请稍后再试。'));
    setScanButtonState('queued');
    startQueuePolling();
    setTimeout(refreshFiles, 500);
    setTimeout(() => setScanButtonState('idle'), 1200);
  } catch {
    setScanButtonState('idle');
  }
}

async function createDemoLibrary() {
  setScanButtonState('loading');
  try {
    const r = await fetch(`${API}/api/demo`, { method: 'POST' });
    if (!r.ok) throw new Error(await responseUserMessage(r, '示例资料导入失败，请稍后再试。'));
    switchView('library');
    setScanButtonState('queued');
    startQueuePolling();
    setTimeout(refreshFiles, 500);
    setTimeout(() => setScanButtonState('idle'), 1200);
  } catch {
    setScanButtonState('idle');
  }
}

async function handleFileSelect(e) {
  await uploadFiles([...e.target.files]);
  e.target.value = '';
}

async function handleDrop(e) {
  e.preventDefault();
  document.getElementById('upload-zone').classList.remove('border-primary/60', 'bg-surface-container');
  const SUPPORTED = ['.pdf','.md','.markdown','.txt','.docx','.py','.rs','.ts','.css','.sh','.jpg','.jpeg','.png','.webp','.heic','.heif'];
  await uploadFiles([...e.dataTransfer.files].filter(f => SUPPORTED.some(ext => f.name.toLowerCase().endsWith(ext))));
}

async function uploadFiles(files) {
  for (const file of files) {
    const fd = new FormData();
    fd.append('file', file);
    try {
      await fetch(`${API}/api/upload`, { method: 'POST', body: fd });
    } catch {}
  }
  startQueuePolling();
  if (document.getElementById('view-library').classList.contains('hidden') === false) {
    refreshFiles();
  }
}

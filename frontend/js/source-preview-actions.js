function selectSourcePreviewChunk(index) {
  sourcePreviewState.selectedIndex = index;
  renderSourcePreview();
}

function openSourceOriginal() {
  const file = sourcePreviewState.file;
  if (file?.id) openFilePreview(file.id);
}

async function savePreviewChunkAsNote() {
  const file = sourcePreviewState.file;
  const chunk = sourcePreviewState.chunks[sourcePreviewState.selectedIndex];
  if (!file || !chunk) return;
  const text = sourceChunkText(chunk);
  if (!text.trim()) return;
  try {
    const r = await fetch(`${API}/api/notes`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        title: `来源片段 - ${file.file_name}`,
        content: `# ${file.file_name}\n\n位置：${chunk.section || ''} ${chunk.page_num ? `第 ${chunk.page_num} 页` : ''}\n\n> ${text}`,
        collection: ['Saved', 'Sources'].join(' '),
        user_tags: ['source', 'citation'],
      }),
    });
    if (!r.ok) throw new Error(await r.text());
    const detail = document.getElementById('source-detail-panel');
    if (detail) detail.insertAdjacentHTML('afterbegin', '<div class="mb-2 rounded-lg bg-tertiary-container px-3 py-2 text-[11px] font-bold text-on-tertiary-container">已保存为笔记。</div>');
    setScanButtonState('queued');
    startQueuePolling();
  } catch (e) {
    const detail = document.getElementById('source-detail-panel');
    if (detail) detail.insertAdjacentHTML('afterbegin', `<div class="mb-2 rounded-lg bg-error/10 px-3 py-2 text-[11px] font-bold text-error">保存失败：${escHtml(e.message)}</div>`);
  }
}

function renderKnowledgeReview(review) {
  const panel = document.getElementById('knowledge-review-panel');
  const count = document.getElementById('knowledge-review-count');
  if (!panel) return;
  const queue = Array.isArray(review?.review_queue) ? review.review_queue : [];
  const recommendations = Array.isArray(review?.recommendations) ? review.recommendations : [];
  const topics = Array.isArray(review?.topic_activity) ? review.topic_activity : [];
  const relationships = Array.isArray(review?.relationship_timeline)
    ? review.relationship_timeline
    : [];
  const signals = review?.signals || {};
  if (count) count.textContent = queue.length ? `${queue.length} 项` : '';
  if (!review) {
    panel.innerHTML = '<div class="rounded-lg bg-surface-container-low px-3 py-3">回顾建议暂时不可用。</div>';
    return;
  }
  const queueMarkup = queue.length ? queue.slice(0, 3).map(item => knowledgeReviewItemMarkup(item)).join('') :
    '<div class="rounded-lg bg-surface-container-low px-3 py-3">导入资料并保存回答后，这里会出现回顾建议。</div>';
  const recommendationMarkup = recommendations.slice(0, 2).map(item => `
    <button ${item.file_id ? `onclick="openFilePreview(${Number(item.file_id)})"` : ''} class="w-full text-left rounded-lg bg-primary-container/70 px-3 py-2 text-xs text-on-surface hover:bg-primary-container transition-colors">
      <div class="font-semibold">${escHtml(item.title || '下一步')}</div>
      <div class="mt-0.5 text-[11px] text-on-surface-variant">${escHtml(item.detail || '')}</div>
    </button>`).join('');
  const relationshipMarkup = relationships.slice(0, 3).map(item => knowledgeRelationshipMarkup(item)).join('');
  const topicMarkup = topics.slice(0, 3).map(topic => `
    <span class="inline-flex items-center rounded-full bg-surface-container-low px-2 py-1 text-[11px] font-semibold text-on-surface-variant">
      ${escHtml(topic.title || '主题')} · ${Number(topic.file_count || 0)}
    </span>`).join('');
  panel.innerHTML = `
    <div class="grid grid-cols-3 gap-2">
      ${knowledgeReviewSignalMarkup('资料', signals.files)}
      ${knowledgeReviewSignalMarkup('问题', signals.questions)}
      ${knowledgeReviewSignalMarkup('关联', Number(signals.backlinks || 0) + Number(signals.source_links || 0))}
    </div>
    <div class="mt-3 flex flex-col gap-2">${queueMarkup}</div>
    ${relationshipMarkup ? `<div class="mt-3 flex flex-col gap-2">${relationshipMarkup}</div>` : ''}
    ${recommendationMarkup ? `<div class="mt-3 flex flex-col gap-2">${recommendationMarkup}</div>` : ''}
    ${topicMarkup ? `<div class="mt-3 flex flex-wrap gap-2">${topicMarkup}</div>` : ''}
  `;
  renderLocalIcons(panel);
}

function knowledgeReviewSignalMarkup(label, value) {
  return `
    <div class="rounded-lg bg-surface-container-low px-3 py-2">
      <div class="text-[11px] text-on-surface-variant/60">${escHtml(label)}</div>
      <div class="mt-0.5 text-sm font-semibold text-on-surface">${Number(value || 0)}</div>
    </div>`;
}

function knowledgeReviewItemMarkup(item) {
  const file = item.file || {};
  const reason = item.reason || '值得回顾';
  const priority = Number(item.priority || 0);
  const keywords = Array.isArray(item.keywords) ? item.keywords.slice(0, 3) : [];
  return `
    <button onclick="openFilePreview(${Number(file.id || 0)})" class="w-full text-left rounded-lg bg-surface-container-low px-3 py-3 hover:bg-surface-container transition-colors">
      <div class="flex items-start justify-between gap-3">
        <div class="min-w-0">
          <div class="font-semibold text-on-surface line-clamp-1">${escHtml(file.file_name || '资料')}</div>
          <div class="mt-1 text-[11px] text-on-surface-variant/60">${escHtml(reason)}</div>
        </div>
        <span class="rounded-full bg-primary-container px-2 py-0.5 text-[11px] font-semibold text-primary">${priority}</span>
      </div>
      ${keywords.length ? `<div class="mt-2 flex flex-wrap gap-1">${keywords.map(word => `<span class="rounded-full bg-surface-container px-2 py-0.5 text-[10px] text-on-surface-variant">${escHtml(word)}</span>`).join('')}</div>` : ''}
    </button>`;
}

function knowledgeRelationshipMarkup(item) {
  const note = item.note || {};
  const source = item.source || {};
  return `
    <button onclick="openFilePreview(${Number(source.id || note.id || 0)})" class="w-full text-left rounded-lg bg-surface-container-low px-3 py-3 hover:bg-surface-container transition-colors">
      <div class="flex items-center gap-2 text-[11px] font-semibold text-primary">
        <span class="material-symbols-outlined" style="font-size:14px">account_tree</span>
        ${escHtml(item.label || '知识关联')}
      </div>
      <div class="mt-1 text-xs text-on-surface line-clamp-1">${escHtml(note.file_name || '保存内容')}</div>
      <div class="mt-0.5 text-[11px] text-on-surface-variant/65 line-clamp-1">来源：${escHtml(source.file_name || '资料')}</div>
    </button>`;
}

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
  const depth = review?.knowledge_depth || {};
  const concepts = Array.isArray(depth.concepts) ? depth.concepts : [];
  const trails = Array.isArray(depth.source_trails) ? depth.source_trails : [];
  const gaps = Array.isArray(depth.coverage_gaps) ? depth.coverage_gaps : [];
  const opportunities = Array.isArray(depth.relationship_opportunities)
    ? depth.relationship_opportunities
    : [];
  const depthActions = Array.isArray(depth.next_actions) ? depth.next_actions : [];
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
  const conceptMarkup = concepts.slice(0, 4).map(item => knowledgeConceptMarkup(item)).join('');
  const trailMarkup = trails.slice(0, 3).map(item => knowledgeSourceTrailMarkup(item)).join('');
  const gapMarkup = gaps.slice(0, 3).map(item => knowledgeCoverageGapMarkup(item)).join('');
  const opportunityMarkup = opportunities.slice(0, 3).map(item => knowledgeRelationshipOpportunityMarkup(item)).join('');
  const depthActionMarkup = depthActions.slice(0, 2).map(item => `
    <button ${item.file_id ? `onclick="openFilePreview(${Number(item.file_id)})"` : ''} class="w-full text-left rounded-lg bg-secondary-container/80 px-3 py-2 text-xs text-on-surface hover:bg-secondary-container transition-colors">
      <div class="font-semibold">${escHtml(item.title || '下一步')}</div>
      <div class="mt-0.5 text-[11px] text-on-surface-variant">${escHtml(item.detail || '')}</div>
    </button>`).join('');
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
    ${conceptMarkup ? `<div class="mt-3">
      <div class="mb-2 text-[11px] font-bold text-on-surface-variant/60">活跃概念</div>
      <div class="flex flex-wrap gap-2">${conceptMarkup}</div>
    </div>` : ''}
    ${gapMarkup ? `<div class="mt-3">
      <div class="mb-2 text-[11px] font-bold text-on-surface-variant/60">待补齐</div>
      <div class="flex flex-col gap-2">${gapMarkup}</div>
    </div>` : ''}
    ${trailMarkup ? `<div class="mt-3">
      <div class="mb-2 text-[11px] font-bold text-on-surface-variant/60">来源轨迹</div>
      <div class="flex flex-col gap-2">${trailMarkup}</div>
    </div>` : ''}
    ${opportunityMarkup ? `<div class="mt-3">
      <div class="mb-2 text-[11px] font-bold text-on-surface-variant/60">可连接资料</div>
      <div class="flex flex-col gap-2">${opportunityMarkup}</div>
    </div>` : ''}
    <div class="mt-3 flex flex-col gap-2">${queueMarkup}</div>
    ${relationshipMarkup ? `<div class="mt-3 flex flex-col gap-2">${relationshipMarkup}</div>` : ''}
    ${depthActionMarkup ? `<div class="mt-3 flex flex-col gap-2">${depthActionMarkup}</div>` : ''}
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

function knowledgeConceptMarkup(item) {
  const file = (item.files || [])[0] || {};
  const fileId = Number(file.id || 0);
  return `
    <button ${fileId ? `onclick="openFilePreview(${fileId})"` : ''} class="rounded-full bg-surface-container-low px-3 py-1.5 text-left text-[11px] font-semibold text-on-surface-variant hover:bg-surface-container transition-colors">
      ${escHtml(item.title || '概念')} · ${Number(item.file_count || 0)} 份资料 · ${Number(item.question_count || 0)} 次提问
    </button>`;
}

function knowledgeSourceTrailMarkup(item) {
  const file = (item.files || [])[0] || {};
  const feedback = item.feedback?.rating === 'useful' ? '已标记有用' :
    item.feedback?.rating === 'not_useful' ? '需要改进' : '未反馈';
  return `
    <button onclick="openFilePreview(${Number(file.id || 0)})" class="w-full text-left rounded-lg bg-surface-container-low px-3 py-3 hover:bg-surface-container transition-colors">
      <div class="flex items-center gap-2 text-[11px] font-semibold text-primary">
        <span class="material-symbols-outlined" style="font-size:14px">route</span>
        问题引用了来源
      </div>
      <div class="mt-1 text-xs text-on-surface line-clamp-1">${escHtml(item.question || '最近问题')}</div>
      <div class="mt-0.5 text-[11px] text-on-surface-variant/65 line-clamp-1">${escHtml(file.file_name || '来源资料')} · ${feedback}</div>
    </button>`;
}

function knowledgeCoverageGapMarkup(item) {
  const file = item.file || {};
  return `
    <button onclick="openFilePreview(${Number(file.id || 0)})" class="w-full text-left rounded-lg border border-outline-variant/70 bg-surface-container-low px-3 py-3 hover:bg-surface-container transition-colors">
      <div class="flex items-start justify-between gap-3">
        <div class="min-w-0">
          <div class="font-semibold text-on-surface line-clamp-1">${escHtml(item.title || '补齐资料')}</div>
          <div class="mt-1 text-[11px] text-on-surface-variant/65">${escHtml(item.detail || '')}</div>
          <div class="mt-1 text-[11px] text-on-surface-variant/55 line-clamp-1">${escHtml(file.file_name || '资料')}</div>
        </div>
        <span class="rounded-full bg-tertiary-container px-2 py-0.5 text-[11px] font-semibold text-tertiary">${Number(item.priority || 0)}</span>
      </div>
    </button>`;
}

function knowledgeRelationshipOpportunityMarkup(item) {
  const source = item.source || {};
  const target = item.target || {};
  const terms = Array.isArray(item.shared_terms) ? item.shared_terms.slice(0, 4) : [];
  return `
    <button onclick="openFilePreview(${Number(source.id || target.id || 0)})" class="w-full text-left rounded-lg bg-surface-container-low px-3 py-3 hover:bg-surface-container transition-colors">
      <div class="flex items-center gap-2 text-[11px] font-semibold text-primary">
        <span class="material-symbols-outlined" style="font-size:14px">hub</span>
        建议建立资料关联
      </div>
      <div class="mt-1 text-xs text-on-surface line-clamp-1">${escHtml(source.file_name || '资料')} ↔ ${escHtml(target.file_name || '资料')}</div>
      <div class="mt-0.5 text-[11px] text-on-surface-variant/65 line-clamp-1">共同线索：${escHtml(terms.join(' · ') || '内容相近')}</div>
    </button>`;
}

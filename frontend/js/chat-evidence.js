function citationEvidencePill(citation) {
  const level = citation?.evidence_level || 'medium';
  const label = citation?.evidence_label || '可核对';
  const classes = {
    strong: 'bg-tertiary-container text-on-tertiary-container',
    medium: 'bg-primary-container text-primary',
    weak: 'bg-error/10 text-error',
  }[level] || 'bg-primary-container text-primary';
  return `<span class="text-[10px] ${classes} px-1.5 py-0.5 rounded font-bold whitespace-nowrap">${escHtml(label)}</span>`;
}

function evidenceSummaryMarkup(evidence) {
  if (!evidence?.label) return '';
  const level = evidence.level || 'medium';
  const tone = {
    strong: 'bg-tertiary-container text-on-tertiary-container',
    medium: 'bg-primary-container text-primary',
    weak: 'bg-error/10 text-error',
    none: 'bg-error/10 text-error',
    conflict: 'bg-error/10 text-error',
  }[level] || 'bg-primary-container text-primary';
  const conflicts = Array.isArray(evidence.conflicts) ? evidence.conflicts : [];
  const recommendations = Array.isArray(evidence.recommendations) ? evidence.recommendations : [];
  return `
    <div class="rounded-xl bg-surface-container-low px-4 py-3 text-xs text-on-surface-variant">
      <div class="flex flex-wrap items-center gap-2">
        <span class="material-symbols-outlined text-primary" style="font-size:15px">verified</span>
        <span class="rounded-lg px-2 py-1 font-bold ${tone}">${escHtml(evidence.label)}</span>
        <span class="font-medium text-on-surface">${escHtml(evidence.summary || '')}</span>
      </div>
      ${conflicts.length ? `<div class="mt-2 grid gap-1">${conflicts.map(item => `
        <div class="rounded-lg bg-error/10 px-3 py-2 text-error">
          ${escHtml(item.message || '来源存在不一致')} ${Array.isArray(item.files) ? escHtml(item.files.join('、')) : ''}
        </div>`).join('')}</div>` : ''}
      ${recommendations.length ? `<div class="mt-2 grid gap-1">${recommendations.map(item => `
        <div class="rounded-lg bg-surface-container px-3 py-2">${escHtml(item)}</div>`).join('')}</div>` : ''}
    </div>`;
}

function renderEvidenceSummary(evidence) {
  const el = document.getElementById('stream-evidence');
  if (!el) return;
  el.innerHTML = evidenceSummaryMarkup(evidence);
  renderLocalIcons(el);
}

function citationEvidenceReason(citation) {
  const reason = citation?.evidence_reason || '';
  const age = citation?.source_age_days;
  if (age === null || age === undefined) return reason;
  return `${reason} 资料约 ${Number(age)} 天前更新。`;
}

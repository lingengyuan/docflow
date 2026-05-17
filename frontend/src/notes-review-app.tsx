import { render } from 'preact';

type RecordMap = Record<string, unknown>;

type DocFlowWindow = Window &
  typeof globalThis & {
    API?: string;
    DocFlowNotesReviewApp?: {
      renderKnowledgeReview: (review: unknown) => void;
      confirmKnowledgeRelationship: (
        sourceId: number,
        targetId: number,
        button?: HTMLButtonElement | null,
      ) => Promise<void>;
    };
    activeLibraryFileId?: number | null;
    loadKnowledgeOverview?: (fileId: number | null) => Promise<unknown>;
    refreshNotesView?: () => Promise<unknown>;
    renderLocalIcons?: (root?: Document | Element) => void;
    responseUserMessage?: (response: Response, fallback: string) => Promise<string>;
    userFacingErrorMessage?: (message: string, fallback: string) => string;
    escHtml?: (value: string) => string;
    openFilePreview?: (fileId: number) => void;
  };

type ReviewItem = {
  file?: RecordMap;
  reason?: string;
  priority?: number;
  keywords?: string[];
};

type WorkflowStep = {
  id: string;
  title: string;
  count: number;
  complete: boolean;
  detail: string;
  next_action?: string;
};

type KnowledgeReviewViewModel = {
  countLabel: string;
  empty: boolean;
  signals: RecordMap;
  workflow: {
    steps: WorkflowStep[];
    completed: number;
    total: number;
    next_step?: RecordMap;
  };
  concepts: RecordMap[];
  trails: RecordMap[];
  gaps: RecordMap[];
  opportunities: RecordMap[];
  queue: ReviewItem[];
  recommendations: RecordMap[];
  relationships: RecordMap[];
  depthActions: RecordMap[];
  topics: RecordMap[];
};

const win = typeof window === 'undefined' ? undefined : (window as DocFlowWindow);

function asRecord(value: unknown): RecordMap {
  return value && typeof value === 'object' && !Array.isArray(value) ? value as RecordMap : {};
}

function asList(value: unknown): RecordMap[] {
  return Array.isArray(value)
    ? value.filter((item): item is RecordMap => Boolean(item) && typeof item === 'object')
    : [];
}

function asReviewItems(value: unknown): ReviewItem[] {
  return asList(value).map(item => ({
    file: asRecord(item.file),
    reason: stringValue(item.reason),
    priority: numberValue(item.priority),
    keywords: Array.isArray(item.keywords) ? item.keywords.map(word => String(word)) : [],
  }));
}

function numberValue(value: unknown): number {
  const parsed = Number(value || 0);
  return Number.isFinite(parsed) ? parsed : 0;
}

function stringValue(value: unknown, fallback = ''): string {
  return typeof value === 'string' && value ? value : fallback;
}

function firstFile(item: RecordMap): RecordMap {
  const files = asList(item.files);
  return files[0] || {};
}

function fileId(file: RecordMap): number {
  return numberValue(file.id);
}

export function buildRelationshipOpportunityAction(item: unknown) {
  const data = asRecord(item);
  const source = asRecord(data.source);
  const target = asRecord(data.target);
  const sourceId = fileId(source);
  const targetId = fileId(target);
  const terms = Array.isArray(data.shared_terms) ? data.shared_terms.slice(0, 4).map(String) : [];
  return {
    source,
    target,
    sourceId,
    targetId,
    previewId: sourceId || targetId,
    canSave: sourceId > 0 && targetId > 0,
    terms,
  };
}

function openFile(file: RecordMap) {
  const id = fileId(file);
  if (id > 0) win?.openFilePreview?.(id);
}

function Icon({ name, className = '', size = '14px' }: {
  name: string;
  className?: string;
  size?: string;
}) {
  return (
    <span class={`material-symbols-outlined ${className}`} style={{ fontSize: size }}>
      {name}
    </span>
  );
}

export function buildKnowledgeReviewViewModel(review: unknown): KnowledgeReviewViewModel {
  const data = asRecord(review);
  const depth = asRecord(data.knowledge_depth);
  const queue = asReviewItems(data.review_queue);
  const workflow = asRecord(data.workflow);
  const steps = asList(workflow.steps).map(step => ({
    id: stringValue(step.id, 'step'),
    title: stringValue(step.title, '步骤'),
    count: numberValue(step.count),
    complete: Boolean(step.complete),
    detail: stringValue(step.detail),
    next_action: stringValue(step.next_action),
  }));
  return {
    countLabel: queue.length ? `${queue.length} 项` : '',
    empty: !review,
    signals: asRecord(data.signals),
    workflow: {
      steps,
      completed: numberValue(workflow.completed),
      total: numberValue(workflow.total || steps.length || 1),
      next_step: asRecord(workflow.next_step),
    },
    concepts: asList(depth.concepts),
    trails: asList(depth.source_trails),
    gaps: asList(depth.coverage_gaps),
    opportunities: asList(depth.relationship_opportunities),
    queue,
    recommendations: asList(data.recommendations),
    relationships: asList(data.relationship_timeline),
    depthActions: asList(depth.next_actions),
    topics: asList(data.topic_activity),
  };
}

function SignalCard({ label, value }: { label: string; value: unknown }) {
  return (
    <div class="rounded-lg bg-surface-container-low px-3 py-2">
      <div class="text-[11px] text-on-surface-variant/60">{label}</div>
      <div class="mt-0.5 text-sm font-semibold text-on-surface">{numberValue(value)}</div>
    </div>
  );
}

function WorkflowCard({ workflow }: { workflow: KnowledgeReviewViewModel['workflow'] }) {
  if (!workflow.steps.length) return null;
  const nextStep = asRecord(workflow.next_step);
  const next = stringValue(
    nextStep.detail,
    stringValue(nextStep.next_action, '继续导入资料并提问'),
  );
  return (
    <section
      class="mt-3 rounded-xl border border-outline-variant/50 bg-surface-container-lowest px-3 py-3"
      aria-label="知识闭环"
    >
      <div class="flex items-start justify-between gap-3">
        <div>
          <div class="text-[11px] font-bold text-on-surface">知识闭环</div>
          <div class="mt-0.5 text-[11px] text-on-surface-variant/65">
            资料、问题、来源、笔记、关联和反馈放在同一条回顾线上。
          </div>
        </div>
        <span class="rounded-full bg-primary-container px-2 py-0.5 text-[11px] font-semibold text-primary">
          {workflow.completed}/{workflow.total}
        </span>
      </div>
      <div class="mt-3 grid grid-cols-2 gap-2">
        {workflow.steps.map(step => (
          <div key={step.id} class="rounded-lg bg-surface-container-low px-3 py-2">
            <div class="flex items-center gap-2">
              <Icon
                name={step.complete ? 'check_circle' : 'radio_button_unchecked'}
                className={`${step.complete ? 'bg-primary text-on-primary' : 'bg-surface-container text-on-surface-variant'} rounded-full`}
              />
              <span class="text-[11px] font-semibold text-on-surface">{step.title}</span>
              <span class="ml-auto text-[11px] text-on-surface-variant/60">{step.count}</span>
            </div>
            <div class="mt-1 text-[10px] leading-relaxed text-on-surface-variant/65">
              {step.detail}
            </div>
          </div>
        ))}
      </div>
      <div class="mt-3 rounded-lg bg-secondary-container/80 px-3 py-2 text-[11px] text-on-surface">
        <span class="font-semibold">下一步：</span>
        {next}
      </div>
    </section>
  );
}

function TopicPills({ topics }: { topics: RecordMap[] }) {
  if (!topics.length) return null;
  return (
    <div class="mt-3 flex flex-wrap gap-2">
      {topics.slice(0, 3).map(topic => (
        <span class="inline-flex items-center rounded-full bg-surface-container-low px-2 py-1 text-[11px] font-semibold text-on-surface-variant">
          {stringValue(topic.title, '主题')} · {numberValue(topic.file_count)}
        </span>
      ))}
    </div>
  );
}

function ConceptList({ concepts }: { concepts: RecordMap[] }) {
  if (!concepts.length) return null;
  return (
    <div class="mt-3">
      <div class="mb-2 text-[11px] font-bold text-on-surface-variant/60">活跃概念</div>
      <div class="flex flex-wrap gap-2">
        {concepts.slice(0, 4).map(item => (
          <button
            onClick={() => openFile(firstFile(item))}
            class="rounded-full bg-surface-container-low px-3 py-1.5 text-left text-[11px] font-semibold text-on-surface-variant hover:bg-surface-container transition-colors"
          >
            {stringValue(item.title, '概念')} · {numberValue(item.file_count)} 份资料 · {numberValue(item.question_count)} 次提问
          </button>
        ))}
      </div>
    </div>
  );
}

function ReviewQueue({ queue }: { queue: ReviewItem[] }) {
  if (!queue.length) {
    return (
      <div class="rounded-lg bg-surface-container-low px-3 py-3">
        导入资料并保存回答后，这里会出现回顾建议。
      </div>
    );
  }
  return (
    <>
      {queue.slice(0, 3).map(item => {
        const file = asRecord(item.file);
        return (
          <button
            onClick={() => openFile(file)}
            class="w-full text-left rounded-lg bg-surface-container-low px-3 py-3 hover:bg-surface-container transition-colors"
          >
            <div class="flex items-start justify-between gap-3">
              <div class="min-w-0">
                <div class="font-semibold text-on-surface line-clamp-1">
                  {stringValue(file.file_name, '资料')}
                </div>
                <div class="mt-1 text-[11px] text-on-surface-variant/60">
                  {item.reason || '值得回顾'}
                </div>
              </div>
              <span class="rounded-full bg-primary-container px-2 py-0.5 text-[11px] font-semibold text-primary">
                {numberValue(item.priority)}
              </span>
            </div>
            {item.keywords?.length ? (
              <div class="mt-2 flex flex-wrap gap-1">
                {item.keywords.slice(0, 3).map(word => (
                  <span class="rounded-full bg-surface-container px-2 py-0.5 text-[10px] text-on-surface-variant">
                    {word}
                  </span>
                ))}
              </div>
            ) : null}
          </button>
        );
      })}
    </>
  );
}

function RelationshipList({ relationships }: { relationships: RecordMap[] }) {
  if (!relationships.length) return null;
  return (
    <div class="mt-3 flex flex-col gap-2">
      {relationships.slice(0, 3).map(item => {
        const note = asRecord(item.note);
        const source = asRecord(item.source);
        return (
          <button
            onClick={() => openFile(fileId(source) ? source : note)}
            class="w-full text-left rounded-lg bg-surface-container-low px-3 py-3 hover:bg-surface-container transition-colors"
          >
            <div class="flex items-center gap-2 text-[11px] font-semibold text-primary">
              <Icon name="account_tree" />
              {stringValue(item.label, '知识关联')}
            </div>
            <div class="mt-1 text-xs text-on-surface line-clamp-1">
              {stringValue(note.file_name, '保存内容')}
            </div>
            <div class="mt-0.5 text-[11px] text-on-surface-variant/65 line-clamp-1">
              来源：{stringValue(source.file_name, '资料')}
            </div>
          </button>
        );
      })}
    </div>
  );
}

function SourceTrailList({ trails }: { trails: RecordMap[] }) {
  if (!trails.length) return null;
  return (
    <div class="mt-3">
      <div class="mb-2 text-[11px] font-bold text-on-surface-variant/60">来源轨迹</div>
      <div class="flex flex-col gap-2">
        {trails.slice(0, 3).map(item => {
          const file = firstFile(item);
          const feedback = asRecord(item.feedback);
          const rating = feedback.rating === 'useful'
            ? '已标记有用'
            : feedback.rating === 'not_useful'
              ? '需要改进'
              : '未反馈';
          return (
            <button
              onClick={() => openFile(file)}
              class="w-full text-left rounded-lg bg-surface-container-low px-3 py-3 hover:bg-surface-container transition-colors"
            >
              <div class="flex items-center gap-2 text-[11px] font-semibold text-primary">
                <Icon name="route" />
                问题引用了来源
              </div>
              <div class="mt-1 text-xs text-on-surface line-clamp-1">
                {stringValue(item.question, '最近问题')}
              </div>
              <div class="mt-0.5 text-[11px] text-on-surface-variant/65 line-clamp-1">
                {stringValue(file.file_name, '来源资料')} · {rating}
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function CoverageGapList({ gaps }: { gaps: RecordMap[] }) {
  if (!gaps.length) return null;
  return (
    <div class="mt-3">
      <div class="mb-2 text-[11px] font-bold text-on-surface-variant/60">待补齐</div>
      <div class="flex flex-col gap-2">
        {gaps.slice(0, 3).map(item => {
          const file = asRecord(item.file);
          return (
            <button
              onClick={() => openFile(file)}
              class="w-full text-left rounded-lg border border-outline-variant/70 bg-surface-container-low px-3 py-3 hover:bg-surface-container transition-colors"
            >
              <div class="flex items-start justify-between gap-3">
                <div class="min-w-0">
                  <div class="font-semibold text-on-surface line-clamp-1">
                    {stringValue(item.title, '补齐资料')}
                  </div>
                  <div class="mt-1 text-[11px] text-on-surface-variant/65">
                    {stringValue(item.detail)}
                  </div>
                  <div class="mt-1 text-[11px] text-on-surface-variant/55 line-clamp-1">
                    {stringValue(file.file_name, '资料')}
                  </div>
                </div>
                <span class="rounded-full bg-tertiary-container px-2 py-0.5 text-[11px] font-semibold text-tertiary">
                  {numberValue(item.priority)}
                </span>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function RelationshipOpportunityList({ opportunities }: { opportunities: RecordMap[] }) {
  if (!opportunities.length) return null;
  return (
    <div class="mt-3">
      <div class="mb-2 text-[11px] font-bold text-on-surface-variant/60">可连接资料</div>
      <div class="flex flex-col gap-2">
        {opportunities.slice(0, 3).map(item => {
          const action = buildRelationshipOpportunityAction(item);
          return (
            <div class="w-full rounded-lg bg-surface-container-low px-3 py-3">
              <div class="flex items-center gap-2 text-[11px] font-semibold text-primary">
                <Icon name="hub" />
                建议建立资料关联
              </div>
              <div class="mt-1 text-xs text-on-surface line-clamp-1">
                {stringValue(action.source.file_name, '资料')} ↔ {stringValue(action.target.file_name, '资料')}
              </div>
              <div class="mt-0.5 text-[11px] text-on-surface-variant/65 line-clamp-1">
                共同线索：{action.terms.join(' · ') || '内容相近'}
              </div>
              <div class="mt-2 flex items-center gap-2">
                <button
                  onClick={() => openFile({ id: action.previewId })}
                  class="toolbar-btn !h-8"
                  title="查看资料"
                  aria-label="查看资料"
                >
                  <Icon name="article" />
                  查看
                </button>
                <button
                  onClick={event => confirmKnowledgeRelationship(
                    action.sourceId,
                    action.targetId,
                    event.currentTarget as HTMLButtonElement,
                  )}
                  disabled={!action.canSave}
                  class="toolbar-btn toolbar-btn-primary !h-8"
                  title="保存关联"
                  aria-label="保存关联"
                >
                  <Icon name="add_link" />
                  保存关联
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function ActionList({ items, tone }: { items: RecordMap[]; tone: 'primary' | 'secondary' }) {
  if (!items.length) return null;
  const className = tone === 'primary'
    ? 'w-full text-left rounded-lg bg-primary-container/70 px-3 py-2 text-xs text-on-surface hover:bg-primary-container transition-colors'
    : 'w-full text-left rounded-lg bg-secondary-container/80 px-3 py-2 text-xs text-on-surface hover:bg-secondary-container transition-colors';
  return (
    <div class="mt-3 flex flex-col gap-2">
      {items.slice(0, tone === 'primary' ? 2 : 2).map(item => (
        <button
          onClick={() => openFile({ id: item.file_id })}
          class={className}
        >
          <div class="font-semibold">{stringValue(item.title, '下一步')}</div>
          <div class="mt-0.5 text-[11px] text-on-surface-variant">
            {stringValue(item.detail)}
          </div>
        </button>
      ))}
    </div>
  );
}

function NotesReviewPanel({ review }: { review: unknown }) {
  const model = buildKnowledgeReviewViewModel(review);
  if (model.empty) {
    return (
      <div class="rounded-lg bg-surface-container-low px-3 py-3">
        回顾建议暂时不可用。
      </div>
    );
  }
  return (
    <>
      <div class="grid grid-cols-3 gap-2">
        <SignalCard label="资料" value={model.signals.files} />
        <SignalCard label="问题" value={model.signals.questions} />
        <SignalCard
          label="关联"
          value={numberValue(model.signals.backlinks) + numberValue(model.signals.source_links)}
        />
      </div>
      <WorkflowCard workflow={model.workflow} />
      <ConceptList concepts={model.concepts} />
      <CoverageGapList gaps={model.gaps} />
      <SourceTrailList trails={model.trails} />
      <RelationshipOpportunityList opportunities={model.opportunities} />
      <div class="mt-3 flex flex-col gap-2">
        <ReviewQueue queue={model.queue} />
      </div>
      <RelationshipList relationships={model.relationships} />
      <ActionList items={model.depthActions} tone="secondary" />
      <ActionList items={model.recommendations} tone="primary" />
      <TopicPills topics={model.topics} />
    </>
  );
}

export async function confirmKnowledgeRelationship(
  sourceId: number,
  targetId: number,
  button?: HTMLButtonElement | null,
) {
  if (!win || !sourceId || !targetId || !button) return;
  const previous = button.innerHTML;
  button.disabled = true;
  button.innerHTML = '<span class="spinner"></span><span class="ml-1.5">保存中…</span>';
  try {
    const response = await fetch(`${win.API || ''}/api/knowledge/relationships`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        source_file_id: sourceId,
        target_file_id: targetId,
        relation: 'manual_relationship',
      }),
    });
    if (!response.ok) {
      const message = await win.responseUserMessage?.(response, '资料关联保存失败，请稍后再试。');
      throw new Error(message || '资料关联保存失败，请稍后再试。');
    }
    button.innerHTML = '<span class="material-symbols-outlined" style="font-size:14px">done</span><span class="ml-1">已保存</span>';
    await Promise.all([
      win.refreshNotesView?.() || Promise.resolve(),
      win.loadKnowledgeOverview?.(win.activeLibraryFileId || null) || Promise.resolve(),
    ]);
  } catch (error) {
    button.disabled = false;
    button.innerHTML = previous;
    const panel = document.getElementById('knowledge-review-panel');
    const raw = error instanceof Error ? error.message : String(error);
    const message = win.userFacingErrorMessage?.(raw, '资料关联保存失败，请稍后再试。')
      || '资料关联保存失败，请稍后再试。';
    const safe = win.escHtml?.(message) || message.replace(/[&<>"']/g, ch => ({
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#39;',
    }[ch] || ch));
    panel?.insertAdjacentHTML(
      'afterbegin',
      `<div class="mb-2 rounded-lg bg-error/10 px-3 py-2 text-[11px] font-bold text-error">保存失败：${safe}</div>`,
    );
  }
}

export function renderKnowledgeReview(review: unknown) {
  if (!win) return;
  const panel = document.getElementById('knowledge-review-panel');
  const count = document.getElementById('knowledge-review-count');
  if (!panel) return;
  const model = buildKnowledgeReviewViewModel(review);
  if (count) count.textContent = model.countLabel;
  render(<NotesReviewPanel review={review} />, panel);
  win.renderLocalIcons?.(panel);
}

if (win) {
  win.DocFlowNotesReviewApp = {
    renderKnowledgeReview,
    confirmKnowledgeRelationship,
  };
}

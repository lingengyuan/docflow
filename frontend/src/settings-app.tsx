import { render } from 'preact';

import { iconSizes, mutedCardClass, panelClass, settingsViewContract } from './design-system';

type DocFlowWindow = Window &
  typeof globalThis & {
    DocFlowSettingsApp?: { mountSettingsView: () => void };
    applyI18n?: () => void;
    renderLocalIcons?: (root?: Document | Element) => void;
    refreshSettings?: () => void;
    toggleHealthPanel?: (event: Event) => void;
    triggerIngest?: () => void;
    switchView?: (view: string) => void;
    toggleLocale?: () => void;
    toggleTheme?: () => void;
  };

const win = window as DocFlowWindow;

function call(action: keyof DocFlowWindow, ...args: unknown[]) {
  return (event: Event) => {
    if (action === 'toggleHealthPanel') {
      win.toggleHealthPanel?.(event);
      return;
    }
    const fn = win[action];
    if (typeof fn === 'function') {
      fn(...args);
    }
  };
}

function Icon({ name, size = iconSizes.sm, className = '' }: {
  name: string;
  size?: string;
  className?: string;
}) {
  return (
    <span class={`material-symbols-outlined ${className}`} style={{ fontSize: size }}>
      {name}
    </span>
  );
}

function SettingsHeader() {
  return (
    <header class="h-16 flex-shrink-0 flex justify-between items-center px-8 bg-surface-container-lowest/90 z-40">
      <div>
        <h1
          id="settings-title"
          class="text-base font-semibold text-on-surface tracking-tight"
          data-i18n="nav.settings"
        >
          设置
        </h1>
        <p class="text-[11px] text-on-surface-variant/60 mt-0.5" data-i18n="settings.subtitle">
          本地状态、模型和资料来源
        </p>
      </div>
      <button
        onClick={call('refreshSettings')}
        title="刷新设置状态"
        aria-label="刷新设置状态"
        class="toolbar-btn whitespace-nowrap"
      >
        <Icon name="sync" size={iconSizes.md} />
        <span class="hidden sm:inline">刷新状态</span>
      </button>
    </header>
  );
}

function HealthPanel() {
  return (
    <section class={panelClass}>
      <div class="flex items-center justify-between gap-3 mb-4">
        <div class="flex items-center gap-2">
          <span id="health-icon" class="w-2 h-2 rounded-full bg-tertiary"></span>
          <h2 class="text-sm font-semibold text-on-surface">系统状态</h2>
        </div>
        <button
          id="health-btn"
          onClick={call('toggleHealthPanel')}
          class="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-surface-container text-xs font-medium text-on-surface-variant hover:bg-surface-container-high active:scale-95 transition-all"
        >
          <span id="health-label">状态</span>
          <Icon name="refresh" size={iconSizes.xs} />
        </button>
      </div>
      <div id="health-panel" class="max-h-[300px] overflow-y-auto custom-scrollbar rounded-lg bg-surface-container-low p-3">
        <div id="health-details" class="grid grid-cols-2 xl:grid-cols-3 gap-2 text-xs text-on-surface-variant"></div>
      </div>
    </section>
  );
}

function SourcesPanel() {
  return (
    <section class={panelClass}>
      <div class="flex items-center gap-2 mb-4">
        <Icon name="folder_managed" size={iconSizes.lg} className="text-primary" />
        <h2 class="text-sm font-semibold text-on-surface">监控目录</h2>
      </div>
      <div id="settings-sources-list" class="flex flex-col gap-2 text-xs text-on-surface-variant"></div>
    </section>
  );
}

function ModelPanel() {
  return (
    <section class={panelClass}>
      <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
        <div>
          <h2 class="text-sm font-semibold text-on-surface">本地模型</h2>
          <p id="llm-status" class="text-[11px] text-on-surface-variant/60 mt-0.5">
            本地模型
          </p>
        </div>
        <div class="relative">
          <button
            id="llm-btn"
            class="flex w-full sm:w-auto items-center justify-between sm:justify-start gap-2 px-3 py-1.5 rounded-lg bg-surface-container text-xs font-medium text-on-surface-variant hover:bg-surface-container-high active:scale-95 transition-all"
          >
            <Icon name="bolt" size={iconSizes.xs} className="text-primary" />
            <span id="llm-current" class="line-clamp-2 text-left">读取中</span>
            <Icon name="keyboard_arrow_down" size={iconSizes.xs} className="text-on-surface-variant" />
          </button>
          <div
            id="llm-dropdown"
            class="hidden absolute right-0 mt-1 bg-surface-container-lowest shadow-xl rounded-xl overflow-hidden z-50 min-w-max"
          ></div>
        </div>
      </div>
      <div id="settings-model-list" class="flex flex-col gap-2 text-xs text-on-surface-variant"></div>
    </section>
  );
}

function PreferencePanel() {
  return (
    <section class={panelClass}>
      <div class="flex items-center gap-2 mb-4">
        <Icon name="tune" size={iconSizes.lg} className="text-primary" />
        <h2 class="text-sm font-semibold text-on-surface">使用偏好</h2>
      </div>
      <div class="grid grid-cols-1 sm:grid-cols-3 gap-2 mb-4">
        <button onClick={call('triggerIngest')} class="toolbar-btn toolbar-btn-primary justify-center">
          <Icon name="folder_sync" />
          扫描文件夹
        </button>
        <button onClick={call('switchView', 'history')} class="toolbar-btn justify-center">
          <Icon name="history" />
          历史记录
        </button>
        <button onClick={call('switchView', 'library')} class="toolbar-btn justify-center">
          <Icon name="folder_open" />
          文件库
        </button>
      </div>
      <PreferenceRow
        title="界面语言"
        detail="切换常用导航和状态文案。"
        labelId="locale-current-label"
        label="中文"
        icon="language"
        onClick={call('toggleLocale')}
        ariaLabel="切换语言"
        titleKey="locale.label"
        detailKey="locale.detail"
        ariaKey="locale.toggle"
      />
      <PreferenceRow
        title="界面主题"
        detail="浅色和深色外观会保持相同的信息层级。"
        labelId="theme-current-label"
        label="浅色"
        icon="lightbulb"
        onClick={call('toggleTheme')}
        buttonId="theme-toggle-btn"
        ariaLabel="切换主题"
        titleKey="theme.label"
        detailKey="theme.detail"
        ariaKey="theme.toggle"
      />
      <div class="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs">
        <div class={mutedCardClass}>
          <div class="font-semibold text-on-surface">默认本地优先</div>
          <div class="mt-1 text-[11px] leading-relaxed text-on-surface-variant/70">
            资料、整理结果和历史都保存在本机。
          </div>
        </div>
        <div class={mutedCardClass}>
          <div class="font-semibold text-on-surface">手动整理</div>
          <div class="mt-1 text-[11px] leading-relaxed text-on-surface-variant/70">
            从文件库统一管理收藏、集合和标签。
          </div>
        </div>
      </div>
    </section>
  );
}

function PreferenceRow(props: {
  title: string;
  detail: string;
  labelId: string;
  label: string;
  icon: string;
  onClick: (event: Event) => void;
  ariaLabel: string;
  titleKey: string;
  detailKey: string;
  ariaKey: string;
  buttonId?: string;
}) {
  return (
    <div class={`${mutedCardClass} mb-3`}>
      <div class="flex items-center justify-between gap-3">
        <div>
          <div class="font-semibold text-on-surface" data-i18n={props.titleKey}>
            {props.title}
          </div>
          <div
            class="mt-1 text-[11px] leading-relaxed text-on-surface-variant/70"
            data-i18n={props.detailKey}
          >
            {props.detail}
          </div>
        </div>
        <button
          id={props.buttonId}
          onClick={props.onClick}
          class="toolbar-btn justify-center"
          aria-label={props.ariaLabel}
          data-i18n-aria={props.ariaKey}
        >
          <Icon name={props.icon} />
          <span id={props.labelId}>{props.label}</span>
        </button>
      </div>
    </div>
  );
}

function ContextPanel() {
  return (
    <aside class="context-panel overflow-y-auto custom-scrollbar p-4">
      <section class="soft-panel p-4">
        <div class="flex items-start justify-between gap-3">
          <div>
            <h2 class="panel-title">状态提示</h2>
            <p class="panel-muted mt-1">基于当前本地服务状态生成。</p>
          </div>
          <button
            onClick={call('refreshSettings')}
            class="icon-button !w-8 !h-8"
            title="刷新状态提示"
            aria-label="刷新状态提示"
          >
            <Icon name="sync" />
          </button>
        </div>
        <div
          id="settings-insights-list"
          role="status"
          aria-live="polite"
          class="mt-3 flex flex-col gap-2 text-xs text-on-surface-variant"
        ></div>
      </section>
      <section class="soft-panel p-4 mt-3">
        <h2 class="panel-title">存储使用</h2>
        <div
          id="settings-storage-list"
          role="status"
          aria-live="polite"
          class="mt-3 flex flex-col gap-2 text-xs text-on-surface-variant"
        >
          <div class="rounded-lg bg-surface-container-low px-3 py-2">正在读取本地存储…</div>
        </div>
      </section>
      <section class="soft-panel p-4 mt-3">
        <h2 class="panel-title">资料范围</h2>
        <div class="mt-3 flex flex-col gap-2 text-xs text-on-surface-variant">
          <div class="rounded-lg bg-surface-container-low px-3 py-2">
            <div class="font-semibold text-on-surface">监控目录</div>
            <div class="mt-1 text-[11px] leading-relaxed text-on-surface-variant/70">
              系统会读取已添加的本地文件夹。
            </div>
          </div>
          <div class="rounded-lg bg-surface-container-low px-3 py-2">
            <div class="font-semibold text-on-surface">采集内容</div>
            <div class="mt-1 text-[11px] leading-relaxed text-on-surface-variant/70">
              网页、临时笔记和知识产物会进入文件库。
            </div>
          </div>
          <button onClick={call('switchView', 'library')} class="toolbar-btn justify-between">
            <span>查看文件库</span>
            <Icon name="arrow_forward" />
          </button>
        </div>
      </section>
    </aside>
  );
}

function SettingsView() {
  return (
    <div
      id={settingsViewContract.viewId}
      class="view hidden flex flex-col flex-1 min-h-0 overflow-hidden"
      role="region"
      aria-labelledby="settings-title"
      tabIndex={-1}
    >
      <SettingsHeader />
      <div class="workspace-shell">
        <div class="workspace-card workspace-grid">
          <section class="overflow-y-auto custom-scrollbar p-5">
            <div class="grid grid-cols-1 xl:grid-cols-2 gap-5 items-start">
              <div class="flex flex-col gap-5">
                <HealthPanel />
                <SourcesPanel />
              </div>
              <div class="flex flex-col gap-5">
                <ModelPanel />
                <PreferencePanel />
              </div>
            </div>
          </section>
          <ContextPanel />
        </div>
      </div>
    </div>
  );
}

export function mountSettingsView() {
  const root = document.getElementById(settingsViewContract.rootId);
  if (!root) return;
  render(<SettingsView />, root);
  win.applyI18n?.();
  win.renderLocalIcons?.(root);
}

win.DocFlowSettingsApp = { mountSettingsView };
mountSettingsView();

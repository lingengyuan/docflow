// ── Lightweight product language support ──
const DOCFLOW_I18N = {
  'zh-CN': {
    'skip.main': '跳到主内容',
    'nav.main': '主导航',
    'nav.chat': '对话',
    'nav.library': '资料库',
    'library.title': '文件库',
    'nav.source': '来源预览',
    'nav.notes': '笔记',
    'nav.settings': '设置',
    'sidebar.storage': '本地存储',
    'sidebar.local': '本地运行',
    'sidebar.privacy': '隐私数据仅在本机处理',
    'state.loading': '正在读取',
    'search.placeholder': '搜索资料库、笔记或提问… (⌘K)',
    'search.aria': '全局搜索或提问',
    'settings.subtitle': '本地状态、模型和资料来源',
    'locale.toggle': '切换语言',
    'locale.label': '界面语言',
    'locale.detail': '切换常用导航和状态文案。',
    'locale.current': '中文',
    'theme.toggle': '切换主题',
    'theme.label': '界面主题',
    'theme.detail': '浅色和深色外观会保持相同的信息层级。',
  },
  en: {
    'skip.main': 'Skip to main content',
    'nav.main': 'Main navigation',
    'nav.chat': 'Chat',
    'nav.library': 'Library',
    'library.title': 'Files',
    'nav.source': 'Sources',
    'nav.notes': 'Notes',
    'nav.settings': 'Settings',
    'sidebar.storage': 'Local storage',
    'sidebar.local': 'Local runtime',
    'sidebar.privacy': 'Private data stays on this device',
    'state.loading': 'Loading',
    'search.placeholder': 'Search library, notes, or ask… (⌘K)',
    'search.aria': 'Search or ask across your library',
    'settings.subtitle': 'Local status, models, and sources',
    'locale.toggle': 'Switch language',
    'locale.label': 'Interface language',
    'locale.detail': 'Switch common navigation and status text.',
    'locale.current': 'English',
    'theme.toggle': 'Switch theme',
    'theme.label': 'Theme',
    'theme.detail': 'Light and dark appearances keep the same information hierarchy.',
  },
};

function activeLocale() {
  return localStorage.getItem('docflow.locale') || 'zh-CN';
}

function t(key) {
  const locale = activeLocale();
  return (DOCFLOW_I18N[locale] && DOCFLOW_I18N[locale][key]) || DOCFLOW_I18N['zh-CN'][key] || key;
}

function applyI18n() {
  const locale = activeLocale();
  document.documentElement.lang = locale;
  document.querySelectorAll('[data-i18n]').forEach(el => {
    el.textContent = t(el.dataset.i18n);
  });
  document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
    el.setAttribute('placeholder', t(el.dataset.i18nPlaceholder));
  });
  document.querySelectorAll('[data-i18n-aria]').forEach(el => {
    el.setAttribute('aria-label', t(el.dataset.i18nAria));
  });
  document.querySelectorAll('[data-i18n-title]').forEach(el => {
    el.setAttribute('title', t(el.dataset.i18nTitle));
  });
  const current = document.getElementById('locale-current-label');
  if (current) current.textContent = t('locale.current');
}

function setLocale(locale) {
  if (!DOCFLOW_I18N[locale]) return;
  localStorage.setItem('docflow.locale', locale);
  if (window.DocFlowState && window.DocFlowState.app) {
    window.DocFlowState.app.locale = locale;
  }
  applyI18n();
}

function toggleLocale() {
  setLocale(activeLocale() === 'zh-CN' ? 'en' : 'zh-CN');
}

document.addEventListener('DOMContentLoaded', applyI18n);

const DOCFLOW_THEMES = {
  light: '浅色',
  dark: '深色',
};

function activeTheme() {
  return localStorage.getItem('docflow.theme') || 'light';
}

function applyTheme() {
  const theme = DOCFLOW_THEMES[activeTheme()] ? activeTheme() : 'light';
  document.documentElement.dataset.theme = theme;
  if (window.DocFlowState?.app) {
    window.DocFlowState.app.theme = theme;
  }
  const label = document.getElementById('theme-current-label');
  if (label) label.textContent = DOCFLOW_THEMES[theme];
}

function setTheme(theme) {
  if (!DOCFLOW_THEMES[theme]) return;
  localStorage.setItem('docflow.theme', theme);
  applyTheme();
}

function toggleTheme() {
  setTheme(activeTheme() === 'dark' ? 'light' : 'dark');
}

applyTheme();

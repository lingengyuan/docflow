export const iconSizes = {
  xs: '14px',
  sm: '15px',
  md: '16px',
  lg: '18px',
} as const;

export const panelClass =
  'rounded-xl bg-surface-container-lowest shadow-sm p-5';

export const mutedCardClass =
  'rounded-lg bg-surface-container-low px-3 py-3';

export const settingsViewContract = {
  rootId: 'settings-view-root',
  viewId: 'view-settings',
  requiredIds: [
    'settings-title',
    'health-icon',
    'health-label',
    'health-details',
    'settings-sources-list',
    'llm-status',
    'llm-btn',
    'llm-current',
    'llm-dropdown',
    'settings-model-list',
    'theme-toggle-btn',
    'settings-insights-list',
    'settings-storage-list',
  ],
} as const;

export type SettingsViewContract = typeof settingsViewContract;

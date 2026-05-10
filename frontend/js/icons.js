const LOCAL_ICON_PATHS = {
  add: '<path d="M12 5v14M5 12h14"/>',
  add_circle: '<circle cx="12" cy="12" r="9"/><path d="M12 8v8M8 12h8"/>',
  add_link: '<path d="M10 13a5 5 0 0 1 0-7l1-1a5 5 0 0 1 7 7l-1 1"/><path d="M14 11a5 5 0 0 1 0 7l-1 1a5 5 0 0 1-7-7l1-1"/><path d="M17 16v5M14.5 18.5h5"/>',
  add_task: '<path d="M5 6h8M5 12h5M5 18h14"/><path d="m13 12 2 2 5-6"/>',
  arrow_back: '<path d="M19 12H5M11 6l-6 6 6 6"/>',
  arrow_forward: '<path d="M5 12h14M13 6l6 6-6 6"/>',
  arrow_upward: '<path d="M12 19V5M6 11l6-6 6 6"/>',
  article: '<path d="M6 3h9l3 3v15H6z"/><path d="M14 3v4h4M9 11h6M9 15h6M9 19h4"/>',
  auto_awesome: '<path class="local-icon-fill" d="m12 2 1.9 5.1L19 9l-5.1 1.9L12 16l-1.9-5.1L5 9l5.1-1.9L12 2Z"/><path d="M5 14v4M3 16h4M18 15v3M16.5 16.5h3"/>',
  bolt: '<path class="local-icon-fill" d="M13 2 5 14h6l-1 8 8-12h-6l1-8Z"/>',
  build: '<path d="m14.7 6.3 3-3a4 4 0 0 1-5 5L5 16a2 2 0 1 0 3 3l7.7-7.7a4 4 0 0 1-1-5Z"/>',
  chat_bubble: '<path d="M5 5h14v10H9l-4 4V5Z"/>',
  check: '<path d="m5 12 4 4L19 6"/>',
  check_box: '<rect x="4" y="4" width="16" height="16" rx="3"/><path d="m8 12 3 3 5-6"/>',
  check_circle: '<circle cx="12" cy="12" r="9"/><path d="m8 12 3 3 5-6"/>',
  close: '<path d="M6 6l12 12M18 6 6 18"/>',
  code: '<path d="m8 9-4 3 4 3M16 9l4 3-4 3M14 5l-4 14"/>',
  content_copy: '<rect x="8" y="8" width="11" height="13" rx="2"/><path d="M5 16H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>',
  delete: '<path d="M4 7h16M10 11v6M14 11v6M6 7l1 14h10l1-14M9 7V4h6v3"/>',
  delete_sweep: '<path d="M4 7h16M10 11v6M14 11v6M6 7l1 14h10l1-14M9 7V4h6v3M3 11h3M2 15h4M1 19h5"/>',
  description: '<path d="M6 3h9l3 3v15H6z"/><path d="M14 3v4h4M9 12h6M9 16h6"/>',
  download: '<path d="M12 3v12M7 10l5 5 5-5M5 21h14"/>',
  edit_note: '<path d="M5 6h10M5 11h8M5 16h6"/><path d="m14 19 5-5 2 2-5 5h-2v-2Z"/>',
  error: '<circle cx="12" cy="12" r="9"/><path d="M12 7v6M12 17h.01"/>',
  filter_alt_off: '<path d="M4 5h16l-6 7v5l-4 2v-7L4 5ZM3 3l18 18"/>',
  folder_managed: '<path d="M3 6h7l2 2h9v10a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2Z"/><circle cx="16" cy="14" r="2"/><path d="M16 10v1M16 17v1M12 14h1M19 14h1"/>',
  folder_open: '<path d="M3 7h7l2 2h9l-2 10H4L3 7Z"/><path d="M3 7v12"/>',
  folder_sync: '<path d="M3 7h7l2 2h9v4"/><path d="M4 19h8"/><path d="M18 14a4 4 0 1 0 2.8 1.2"/><path d="M21 14v3h-3"/>',
  format_quote: '<path d="M7 7h5v5H9c0 2 1 3 3 4v2c-4-1-6-4-6-8V7ZM15 7h5v5h-3c0 2 1 3 3 4v2c-4-1-6-4-6-8V7Z"/>',
  forum: '<path d="M4 5h12v8H8l-4 4V5Z"/><path d="M10 15h6l4 4V9h-2"/>',
  history: '<path d="M4 12a8 8 0 1 0 3-6"/><path d="M4 4v5h5M12 8v5l3 2"/>',
  image: '<rect x="4" y="5" width="16" height="14" rx="2"/><circle cx="9" cy="10" r="2"/><path d="m4 17 4-4 3 3 3-4 6 5"/>',
  inventory_2: '<path d="M4 7 12 3l8 4-8 4-8-4Z"/><path d="M4 7v10l8 4 8-4V7M12 11v10"/>',
  keyboard_arrow_down: '<path d="m6 9 6 6 6-6"/>',
  language: '<circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3a14 14 0 0 1 0 18M12 3a14 14 0 0 0 0 18"/>',
  lightbulb: '<path d="M9 18h6M10 22h4M8 14a6 6 0 1 1 8 0c-1 1-1 2-1 4H9c0-2 0-3-1-4Z"/>',
  link: '<path d="M10 13a5 5 0 0 1 0-7l1-1a5 5 0 0 1 7 7l-1 1"/><path d="M14 11a5 5 0 0 1 0 7l-1 1a5 5 0 0 1-7-7l1-1"/>',
  note_add: '<path d="M6 3h9l3 3v15H6z"/><path d="M14 3v4h4M12 11v6M9 14h6"/>',
  open_in_new: '<path d="M14 4h6v6M20 4l-9 9"/><path d="M20 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V5a1 1 0 0 1 1-1h5"/>',
  schedule: '<circle cx="12" cy="12" r="9"/><path d="M12 7v6l4 2"/>',
  sell: '<path d="M4 12V5h7l9 9-7 7-9-9Z"/><circle cx="8" cy="8" r="1"/>',
  search: '<circle cx="11" cy="11" r="7"/><path d="m16 16 4 4"/>',
  table: '<rect x="4" y="5" width="16" height="14" rx="2"/><path d="M4 10h16M4 15h16M10 5v14M15 5v14"/>',
  star: '<path class="local-icon-fill" d="m12 3 2.8 5.7 6.2.9-4.5 4.4 1.1 6.2L12 17.3 6.4 20.2 7.5 14 3 9.6l6.2-.9L12 3Z"/>',
  summarize: '<path d="M6 3h9l3 3v15H6z"/><path d="M14 3v4h4M9 11h6M9 15h4M9 19h6"/>',
  sync: '<path d="M20 7v5h-5"/><path d="M4 17v-5h5"/><path d="M6 9a7 7 0 0 1 11.5-2M18 15a7 7 0 0 1-11.5 2"/>',
  tune: '<path d="M4 7h10M18 7h2M4 12h3M11 12h9M4 17h8M16 17h4"/><circle cx="16" cy="7" r="2"/><circle cx="9" cy="12" r="2"/><circle cx="14" cy="17" r="2"/>',
  upload_file: '<path d="M6 3h9l3 3v15H6z"/><path d="M14 3v4h4M12 18V10M8 14l4-4 4 4"/>',
  visibility: '<path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12Z"/><circle cx="12" cy="12" r="3"/>',
  warning: '<path d="M12 3 2 21h20L12 3Z"/><path d="M12 9v5M12 18h.01"/>',
};

const LOCAL_ICON_ALIASES = {
  autorenew: 'sync',
  refresh: 'sync',
  task_alt: 'check_circle',
};

function iconSvg(token) {
  const name = LOCAL_ICON_ALIASES[token] || token;
  const paths = LOCAL_ICON_PATHS[name] || '<circle cx="12" cy="12" r="8"/><path d="M12 8v4M12 16h.01"/>';
  return `<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">${paths}</svg>`;
}

function getIconToken(icon) {
  return icon?.dataset?.icon || icon?.textContent?.trim() || '';
}

function setIcon(icon, token) {
  if (!icon || !token) return;
  icon.dataset.icon = token;
  icon.setAttribute('aria-hidden', 'true');
  icon.innerHTML = iconSvg(token);
}

function renderLocalIcons(root = document) {
  const icons = root.matches?.('.material-symbols-outlined')
    ? [root]
    : Array.from(root.querySelectorAll?.('.material-symbols-outlined') || []);
  icons.forEach(icon => setIcon(icon, getIconToken(icon)));
}

function initLocalIcons() {
  renderLocalIcons(document);
  const observer = new MutationObserver(mutations => {
    for (const mutation of mutations) {
      mutation.addedNodes.forEach(node => {
        if (node.nodeType === Node.ELEMENT_NODE) renderLocalIcons(node);
      });
    }
  });
  observer.observe(document.body, { childList: true, subtree: true });
}

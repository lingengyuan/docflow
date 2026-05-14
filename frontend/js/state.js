// ── Shared application state ──
window.API = '';

window.DocFlowState = {
  app: {
    pendingConfirmResolve: null,
    locale: localStorage.getItem('docflow.locale') || 'zh-CN',
    theme: localStorage.getItem('docflow.theme') || 'light',
  },
  chat: {
    currentConversationId: null,
    conversationItems: [],
    lastCitations: [],
    lastRelatedNotes: [],
    lastHistoryId: null,
    chatPreviewCitation: null,
    queryScopeFiles: [],
  },
  library: {
    selectedFileIds: new Set(),
    favoritedIds: new Set(),
    files: [],
    meta: { collections: [], user_tags: [] },
    filters: { status: '', collection: '', tag: '', favorite: false, kind: '', recent: false },
    page: 1,
    refreshRequestId: 0,
    workflowMode: 'note',
    activeFileId: null,
    contextTab: 'details',
    sourceReview: { fileId: null, chunks: [], status: 'idle', error: '', requestId: 0, file: null },
  },
  notes: {
    knowledgeSourceFileIds: [],
  },
  source: {
    previewState: { file: null, chunks: [], selectedIndex: 0, loading: false, error: '' },
  },
  settings: {
    healthSnapshot: null,
    llmOptions: [],
    storageUsage: null,
  },
  queue: {
    pollTimer: null,
    filesRefreshKey: '',
  },
};

Object.defineProperty(window, 'libraryPageSize', { value: 14, writable: false });

Object.defineProperties(window, {
  currentConversationId: { get: () => DocFlowState.chat.currentConversationId, set: value => { DocFlowState.chat.currentConversationId = value; } },
  conversationItems: { get: () => DocFlowState.chat.conversationItems, set: value => { DocFlowState.chat.conversationItems = value; } },
  lastCitations: { get: () => DocFlowState.chat.lastCitations, set: value => { DocFlowState.chat.lastCitations = value; } },
  lastRelatedNotes: { get: () => DocFlowState.chat.lastRelatedNotes, set: value => { DocFlowState.chat.lastRelatedNotes = value; } },
  lastHistoryId: { get: () => DocFlowState.chat.lastHistoryId, set: value => { DocFlowState.chat.lastHistoryId = value; } },
  chatPreviewCitation: { get: () => DocFlowState.chat.chatPreviewCitation, set: value => { DocFlowState.chat.chatPreviewCitation = value; } },
  queryScopeFiles: { get: () => DocFlowState.chat.queryScopeFiles, set: value => { DocFlowState.chat.queryScopeFiles = value; } },

  selectedFileIds: { get: () => DocFlowState.library.selectedFileIds, set: value => { DocFlowState.library.selectedFileIds = value; } },
  favoritedIds: { get: () => DocFlowState.library.favoritedIds, set: value => { DocFlowState.library.favoritedIds = value; } },
  libraryFiles: { get: () => DocFlowState.library.files, set: value => { DocFlowState.library.files = value; } },
  libraryMeta: { get: () => DocFlowState.library.meta, set: value => { DocFlowState.library.meta = value; } },
  libraryFilters: { get: () => DocFlowState.library.filters, set: value => { DocFlowState.library.filters = value; } },
  libraryPage: { get: () => DocFlowState.library.page, set: value => { DocFlowState.library.page = value; } },
  refreshFilesRequestId: { get: () => DocFlowState.library.refreshRequestId, set: value => { DocFlowState.library.refreshRequestId = value; } },
  libraryWorkflowMode: { get: () => DocFlowState.library.workflowMode, set: value => { DocFlowState.library.workflowMode = value; } },
  activeLibraryFileId: { get: () => DocFlowState.library.activeFileId, set: value => { DocFlowState.library.activeFileId = value; } },
  libraryContextTab: { get: () => DocFlowState.library.contextTab, set: value => { DocFlowState.library.contextTab = value; } },
  librarySourceReview: { get: () => DocFlowState.library.sourceReview, set: value => { DocFlowState.library.sourceReview = value; } },

  knowledgeSourceFileIds: { get: () => DocFlowState.notes.knowledgeSourceFileIds, set: value => { DocFlowState.notes.knowledgeSourceFileIds = value; } },
  sourcePreviewState: { get: () => DocFlowState.source.previewState, set: value => { DocFlowState.source.previewState = value; } },

  healthSnapshot: { get: () => DocFlowState.settings.healthSnapshot, set: value => { DocFlowState.settings.healthSnapshot = value; } },
  llmOptions: { get: () => DocFlowState.settings.llmOptions, set: value => { DocFlowState.settings.llmOptions = value; } },
  storageUsage: { get: () => DocFlowState.settings.storageUsage, set: value => { DocFlowState.settings.storageUsage = value; } },

  queuePollTimer: { get: () => DocFlowState.queue.pollTimer, set: value => { DocFlowState.queue.pollTimer = value; } },
  queuePollTimer2: { get: () => DocFlowState.queue.pollTimer, set: value => { DocFlowState.queue.pollTimer = value; } },
  lastQueueFilesRefreshKey: { get: () => DocFlowState.queue.filesRefreshKey, set: value => { DocFlowState.queue.filesRefreshKey = value; } },

  pendingConfirmResolve: { get: () => DocFlowState.app.pendingConfirmResolve, set: value => { DocFlowState.app.pendingConfirmResolve = value; } },
  docflowLocale: { get: () => DocFlowState.app.locale, set: value => { DocFlowState.app.locale = value; } },
  docflowTheme: { get: () => DocFlowState.app.theme, set: value => { DocFlowState.app.theme = value; } },
});

const {
  ItemView,
  MarkdownView,
  Notice,
  Plugin,
  PluginSettingTab,
  Setting,
  requestUrl,
} = require('obsidian');

const VIEW_TYPE_DOCFLOW = 'docflow-assistant-view';
const DEFAULT_SETTINGS = {
  docflowUrl: 'http://127.0.0.1:8000',
};

module.exports = class DocFlowAssistantPlugin extends Plugin {
  async onload() {
    await this.loadSettings();
    this.lastCitations = [];

    this.registerView(
      VIEW_TYPE_DOCFLOW,
      leaf => new DocFlowAssistantView(leaf, this),
    );

    this.addRibbonIcon('sparkles', 'DocFlow Assistant', () => this.activateView());
    this.addCommand({
      id: 'open-docflow-assistant',
      name: 'Open DocFlow assistant',
      callback: () => this.activateView(),
    });
    this.addCommand({
      id: 'ask-docflow-selection',
      name: 'Ask DocFlow with selected text',
      editorCallback: async editor => {
        const selected = editor.getSelection().trim();
        if (!selected) {
          new Notice('Select text first.');
          return;
        }
        const view = await this.activateView();
        view.setQuestion(selected);
        await view.ask(selected);
      },
    });
    this.addCommand({
      id: 'find-docflow-related-notes',
      name: 'Find DocFlow related notes for current note',
      callback: async () => {
        const view = await this.activateView();
        await view.findRelated(this.currentNotePayload());
      },
    });
    this.addCommand({
      id: 'insert-docflow-citations',
      name: 'Insert last DocFlow citations',
      editorCallback: editor => this.insertLastCitations(editor),
    });

    this.addSettingTab(new DocFlowSettingTab(this.app, this));
  }

  onunload() {
    this.app.workspace.detachLeavesOfType(VIEW_TYPE_DOCFLOW);
  }

  async loadSettings() {
    this.settings = Object.assign({}, DEFAULT_SETTINGS, await this.loadData());
  }

  async saveSettings() {
    await this.saveData(this.settings);
  }

  async activateView() {
    let leaf = this.app.workspace.getLeavesOfType(VIEW_TYPE_DOCFLOW)[0];
    if (!leaf) {
      leaf = this.app.workspace.getRightLeaf(false);
      await leaf.setViewState({ type: VIEW_TYPE_DOCFLOW, active: true });
    }
    this.app.workspace.revealLeaf(leaf);
    return leaf.view;
  }

  async postJson(path, payload) {
    const base = this.settings.docflowUrl.replace(/\/+$/, '');
    const response = await requestUrl({
      url: `${base}${path}`,
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    return response.json;
  }

  currentNotePayload() {
    const view = this.app.workspace.getActiveViewOfType(MarkdownView);
    const file = view?.file;
    return {
      note_title: file?.basename || '',
      note_path: file?.path || '',
      note_content: view?.editor?.getValue()?.slice(0, 12000) || '',
      selection: view?.editor?.getSelection() || '',
    };
  }

  insertLastCitations(editor) {
    if (!this.lastCitations.length) {
      new Notice('No DocFlow citations to insert.');
      return;
    }
    const markdown = [
      '',
      '## DocFlow citations',
      ...this.lastCitations.map(citation => {
        const page = citation.page_num ? ` p.${citation.page_num}` : '';
        const section = citation.section ? ` - ${citation.section}` : '';
        return `- [[${citation.file_name || 'source'}]]${page}${section}`;
      }),
      '',
    ].join('\n');
    editor.replaceSelection(markdown);
    new Notice('DocFlow citations inserted.');
  }
};

class DocFlowAssistantView extends ItemView {
  constructor(leaf, plugin) {
    super(leaf);
    this.plugin = plugin;
    this.answer = '';
    this.citations = [];
    this.relatedNotes = [];
  }

  getViewType() {
    return VIEW_TYPE_DOCFLOW;
  }

  getDisplayText() {
    return 'DocFlow';
  }

  getIcon() {
    return 'sparkles';
  }

  async onOpen() {
    this.render();
  }

  setQuestion(value) {
    const input = this.containerEl.querySelector('.docflow-question');
    if (input) input.value = value;
  }

  render() {
    const root = this.contentEl;
    root.empty();
    root.addClass('docflow-assistant');

    const title = root.createEl('h2', { text: 'DocFlow' });
    title.addClass('docflow-title');

    const input = root.createEl('textarea', {
      cls: 'docflow-question',
      attr: { placeholder: 'Ask your local knowledge base...' },
    });

    const actions = root.createDiv({ cls: 'docflow-actions' });
    actions.createEl('button', { text: 'Ask' }).onclick = () => this.ask(input.value);
    actions.createEl('button', { text: 'Use selection' }).onclick = async () => {
      const selected = this.plugin.currentNotePayload().selection;
      if (!selected.trim()) {
        new Notice('Select text first.');
        return;
      }
      input.value = selected;
      await this.ask(selected);
    };
    actions.createEl('button', { text: 'Related notes' }).onclick = () => this.findRelated(this.plugin.currentNotePayload());

    this.statusEl = root.createDiv({ cls: 'docflow-status', text: 'Ready.' });
    this.answerEl = root.createDiv({ cls: 'docflow-answer' });
    this.citationsEl = root.createDiv({ cls: 'docflow-list' });
    this.relatedEl = root.createDiv({ cls: 'docflow-list' });
  }

  async ask(question) {
    const text = String(question || '').trim();
    if (!text) return;
    this.setStatus('Asking DocFlow...');
    try {
      const result = await this.plugin.postJson('/api/query', {
        question: text,
        scope_mode: 'all',
      });
      this.answer = result.answer || '';
      this.citations = result.citations || [];
      this.relatedNotes = result.related_notes || [];
      this.plugin.lastCitations = this.citations;
      this.renderAnswer();
      this.setStatus('Answer ready.');
    } catch (error) {
      this.setStatus(`DocFlow request failed: ${error.message || error}`);
    }
  }

  async findRelated(payload) {
    this.setStatus('Finding related notes...');
    try {
      const result = await this.plugin.postJson('/api/obsidian/related', payload);
      this.relatedNotes = result.related_notes || [];
      this.renderRelated();
      this.setStatus(`${this.relatedNotes.length} related notes found.`);
    } catch (error) {
      this.setStatus(`Related notes failed: ${error.message || error}`);
    }
  }

  renderAnswer() {
    this.answerEl.empty();
    this.answerEl.createEl('h3', { text: 'Answer' });
    this.answerEl.createEl('p', { text: this.answer || 'No answer yet.' });
    this.renderCitations();
    this.renderRelated();
  }

  renderCitations() {
    this.citationsEl.empty();
    this.citationsEl.createEl('h3', { text: 'Citations' });
    if (!this.citations.length) {
      this.citationsEl.createEl('p', { text: 'No citations yet.' });
      return;
    }
    const list = this.citationsEl.createEl('ul');
    for (const citation of this.citations) {
      list.createEl('li', {
        text: `${citation.file_name || 'source'}${citation.page_num ? ` p.${citation.page_num}` : ''}`,
      });
    }
  }

  renderRelated() {
    this.relatedEl.empty();
    this.relatedEl.createEl('h3', { text: 'Related notes' });
    if (!this.relatedNotes.length) {
      this.relatedEl.createEl('p', { text: 'No related notes yet.' });
      return;
    }
    const list = this.relatedEl.createEl('ul');
    for (const note of this.relatedNotes) {
      const item = list.createEl('li');
      item.createEl('strong', { text: note.file_name || 'Untitled' });
      if (note.snippet) item.createEl('p', { text: note.snippet });
    }
  }

  setStatus(message) {
    if (this.statusEl) this.statusEl.setText(message);
  }
}

class DocFlowSettingTab extends PluginSettingTab {
  constructor(app, plugin) {
    super(app, plugin);
    this.plugin = plugin;
  }

  display() {
    const { containerEl } = this;
    containerEl.empty();
    containerEl.createEl('h2', { text: 'DocFlow Assistant' });
    new Setting(containerEl)
      .setName('DocFlow URL')
      .setDesc('Local DocFlow server URL.')
      .addText(text => text
        .setPlaceholder(DEFAULT_SETTINGS.docflowUrl)
        .setValue(this.plugin.settings.docflowUrl)
        .onChange(async value => {
          this.plugin.settings.docflowUrl = value.trim() || DEFAULT_SETTINGS.docflowUrl;
          await this.plugin.saveSettings();
        }));
  }
}

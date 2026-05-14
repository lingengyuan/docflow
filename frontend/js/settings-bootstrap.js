initLocalIcons();
loadLLMOptions();
loadConversations();
loadQueryScopeOptions();
loadLatestAnswerPreview();
syncKnowledgeOutputCards();

const llmButton = document.getElementById('llm-btn');
if (llmButton) {
  llmButton.addEventListener('click', event => {
    event.stopPropagation();
    document.getElementById('llm-dropdown')?.classList.toggle('hidden');
  });
}

document.addEventListener('click', () => {
  document.getElementById('llm-dropdown')?.classList.add('hidden');
  document.getElementById('conversation-menu')?.classList.add('hidden');
});

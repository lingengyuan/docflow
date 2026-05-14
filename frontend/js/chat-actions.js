function copyTextFromButton(btn) {
  const text = btn.dataset.copyText || '';
  if (text) {
    navigator.clipboard.writeText(text).then(() => {
      const icon = btn.querySelector('.material-symbols-outlined');
      setIcon(icon, 'check');
      setTimeout(() => { setIcon(icon, 'content_copy'); }, 1500);
    });
  }
}

function exportAnswerFromButton(btn) {
  const text = btn.dataset.answerText || '';
  if (!text) return;
  const blob = new Blob([text], { type: 'text/markdown;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `docflow-answer-${new Date().toISOString().slice(0,10)}.md`;
  a.click();
  URL.revokeObjectURL(url);
}

async function saveAnswerFromButton(btn) {
  const answer = btn.dataset.answerText || '';
  if (!answer.trim()) return;
  let citations = [];
  try {
    citations = JSON.parse(decodeURIComponent(btn.dataset.citationsJson || '[]'));
  } catch {
    citations = [];
  }
  btn.disabled = true;
  const icon = btn.querySelector('.material-symbols-outlined');
  const previousIcon = getIconToken(icon);
  setIcon(icon, 'sync');
  icon.classList.add('animate-spin');
  try {
    const r = await fetch(`${API}/api/notes/from-answer`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        title: `已保存回答 ${new Date().toISOString().slice(0,10)}`,
        question: btn.dataset.questionText || '',
        answer,
        citations,
        collection: ['Saved', 'Answers'].join(' '),
        user_tags: ['answer'],
      }),
    });
    if (!r.ok) throw new Error(await r.text());
    icon.classList.remove('animate-spin');
    setIcon(icon, 'check');
    setTimeout(() => { setIcon(icon, previousIcon); }, 1500);
    if (!document.getElementById('view-library').classList.contains('hidden')) refreshFiles();
  } catch (e) {
    icon.classList.remove('animate-spin');
    setIcon(icon, 'error');
    alert(`保存笔记失败：${e.message}`);
    setTimeout(() => { setIcon(icon, previousIcon); }, 1500);
  } finally {
    btn.disabled = false;
  }
}

async function sendMessage() {
  const input = document.getElementById('input');
  const question = input.value.trim();
  if (!question) return;
  const scopePayload = buildQueryScopePayload();
  const scopeError = validateQueryScopePayload(scopePayload);
  if (scopeError) {
    alert(scopeError);
    return;
  }

  input.value = '';
  input.style.height = 'auto';
  document.getElementById('send-btn').disabled = true;

  appendUserMessage(question);
  appendThinking();
  const startedAt = performance.now();

  try {
    const body = { question, conversation_id: currentConversationId, ...scopePayload };
    const r = await fetch(`${API}/api/query/stream`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body),
    });
    if (!r.ok || !r.body) {
      throw new Error(await responseUserMessage(r, '本次查询失败，请稍后再试。'));
    }

    document.getElementById('thinking-indicator')?.remove();
    const msgContainer = createAIMessageContainer(question);
    const msgs = document.getElementById('messages');
    const prose = document.getElementById('stream-prose');
    const meta = document.getElementById('stream-meta');

    let answerText = '';
    const reader = r.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let streamCompleted = false;
    let streamErrored = false;
    let relatedNotes = [];
    let historyId = null;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const parsed = DocFlowStreamParser.consumeSseBuffer(buffer);
      buffer = parsed.buffer;

      for (const { event: eventType, data: eventData } of parsed.events) {
        if (eventType === 'conversation') {
          const payload = JSON.parse(eventData);
          currentConversationId = payload.conversation_id;
        } else if (eventType === 'citations') {
          const citations = JSON.parse(eventData);
          renderCitations(citations);
          const saveBtn = msgContainer.querySelector('.answer-save');
          if (saveBtn) saveBtn.dataset.citationsJson = encodeURIComponent(JSON.stringify(citations));
        } else if (eventType === 'evidence') {
          renderEvidenceSummary(JSON.parse(eventData || '{}'));
        } else if (eventType === 'quality') {
          renderAnswerQuality(JSON.parse(eventData || '{}'));
        } else if (eventType === 'related_notes') {
          relatedNotes = JSON.parse(eventData);
          renderRelatedNotes(relatedNotes);
          const relatedEl = document.getElementById('stream-related-notes');
          if (relatedEl) {
            relatedEl.innerHTML = relatedNotes.length ? `<div class="rounded-xl bg-surface-container-low px-4 py-3">
              <div class="mb-2 flex items-center gap-2 text-xs font-bold text-on-surface">
                <span class="material-symbols-outlined text-primary" style="font-size:15px">hub</span>相关笔记
              </div>
              <div class="grid gap-2">${relatedNotesMarkup(relatedNotes)}</div>
            </div>` : '';
            renderLocalIcons(relatedEl);
          }
        } else if (eventType === 'token') {
          const token = JSON.parse(eventData);
          answerText += token;
          prose.classList.add('streaming-cursor');
          prose.innerHTML = renderMarkdown(answerText);
          msgs.scrollTop = msgs.scrollHeight;
        } else if (eventType === 'done') {
          streamCompleted = true;
          const payload = JSON.parse(eventData || '{}');
          historyId = Number(payload?.history_id || 0) || null;
          lastHistoryId = historyId;
          prose.classList.remove('streaming-cursor');
          const elapsedMs = performance.now() - startedAt;
          if (meta) meta.textContent = `耗时 ${(elapsedMs / 1000).toFixed(1)} 秒`;
          msgContainer.querySelector('.answer-copy').dataset.copyText = answerText;
          msgContainer.querySelector('.answer-save').dataset.answerText = answerText;
          msgContainer.querySelector('.answer-export').dataset.answerText = answerText;
          const feedback = msgContainer.querySelector('.answer-feedback');
          if (feedback) feedback.outerHTML = feedbackControlsMarkup(historyId);
          loadConversations();
        } else if (eventType === 'error') {
          streamErrored = true;
          prose.classList.remove('streaming-cursor');
          let rawError = eventData;
          try { rawError = JSON.parse(eventData); } catch {}
          prose.innerHTML = `<span class="text-error">${escHtml(userFacingErrorMessage(rawError, '本次回答失败，请稍后再试。'))}</span>`;
          if (meta) meta.textContent = '回答失败';
        }
      }
    }

    prose.classList.remove('streaming-cursor');
    if (!streamCompleted && !streamErrored) {
      const interrupted = '连接已中断，以上内容可能不完整。';
      answerText = answerText ? `${answerText}\n\n${interrupted}` : interrupted;
      prose.innerHTML = renderMarkdown(answerText);
      msgContainer.querySelector('.answer-copy').dataset.copyText = answerText;
      msgContainer.querySelector('.answer-save').dataset.answerText = answerText;
      msgContainer.querySelector('.answer-export').dataset.answerText = answerText;
      if (meta) meta.textContent = '连接中断';
    }
    prose.removeAttribute('id');
    document.getElementById('stream-citations')?.removeAttribute('id');
    document.getElementById('stream-evidence')?.removeAttribute('id');
    document.getElementById('stream-quality')?.removeAttribute('id');
    document.getElementById('stream-related-notes')?.removeAttribute('id');
  } catch (e) {
    document.getElementById('thinking-indicator')?.remove();
    const msgs = document.getElementById('messages');
    const inner = msgs.querySelector('.max-w-2xl');
    const div = document.createElement('div');
    div.className = 'rounded-xl bg-surface-container-low border border-error/20 px-4 py-3 text-sm text-error';
    div.textContent = userFacingErrorMessage(e.message, '本次查询失败，请稍后再试。');
    inner.appendChild(div);
  } finally {
    document.getElementById('send-btn').disabled = false;
    input.focus();
  }
}

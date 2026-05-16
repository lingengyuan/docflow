"""Accessibility and answer-state browser acceptance checks."""

from __future__ import annotations

from typing import Any


def check_basic_accessibility_contract(page: Any) -> dict[str, int]:
    violations = page.evaluate(
        """
        () => {
          const hasAssociatedLabel = el => {
            if (el.getAttribute('aria-label') || el.getAttribute('aria-labelledby')) return true;
            if (el.id && document.querySelector(`label[for="${CSS.escape(el.id)}"]`)) return true;
            return Boolean(el.closest('label'));
          };
          const unlabeledButtons = Array.from(document.querySelectorAll('button,[role="button"]'))
            .filter(el => {
              const text = (el.innerText || '').trim();
              return !text && !el.getAttribute('aria-label') && !el.getAttribute('title');
            })
            .map(el => el.id || el.className || el.tagName);
          const unlabeledFields = Array.from(document.querySelectorAll(
            'input:not([type="hidden"]), textarea, select'
          ))
            .filter(el => {
              const type = (el.getAttribute('type') || '').toLowerCase();
              if (['button', 'submit', 'file'].includes(type)) return false;
              if (el.offsetParent === null) return false;
              return !hasAssociatedLabel(el);
            })
            .map(el => el.id || el.tagName);
          const missingLiveRegions = [
            'query-scope-status',
            'queue-banner',
            'workflow-status',
            'notes-status',
            'notes-url-status',
            'knowledge-status',
            'settings-insights-list',
            'settings-storage-list',
            'chat-context-queue',
          ]
            .filter(id => !document.getElementById(id)?.getAttribute('aria-live'));
          return { unlabeledButtons, unlabeledFields, missingLiveRegions };
        }
        """
    )
    problems = [
        f"unlabeled buttons: {violations['unlabeledButtons']}",
        f"unlabeled fields: {violations['unlabeledFields']}",
        f"missing live regions: {violations['missingLiveRegions']}",
    ]
    if any(violations.values()):
        raise AssertionError("; ".join(problems))
    return {"checked": 3}


def check_status_messages_are_announced(page: Any) -> dict[str, int]:
    result = page.evaluate(
        """
        () => {
          const ids = [
            'query-scope-status',
            'queue-banner',
            'workflow-status',
            'notes-status',
            'notes-url-status',
            'knowledge-status',
            'settings-insights-list',
            'settings-storage-list',
            'chat-context-queue',
          ];
          const missing = ids.filter(id => {
            const el = document.getElementById(id);
            return !el || el.getAttribute('role') !== 'status' ||
              el.getAttribute('aria-live') !== 'polite';
          });
          const developerTerms = [
            'python main.py',
            'restore-drill',
            'repair-ids',
            'browser-acceptance',
            'doctor',
            'dry-run',
          ];
          const visibleText = document.body.innerText || '';
          const leaked = developerTerms.filter(term => visibleText.includes(term));
          return { missing, leaked, checked: ids.length };
        }
        """
    )
    if result["missing"] or result["leaked"]:
        raise AssertionError(
            f"status regions missing={result['missing']} leaked={result['leaked']}"
        )
    return {"checked": result["checked"]}


def check_answer_quality_states_render(page: Any) -> dict[str, int]:
    text = page.evaluate(
        """
        () => {
          const cases = [
            {
              status: 'grounded',
              label: '已基于本地资料回答',
              reason: '回答由检索到的本地资料支持。',
              answer_mode: 'generated',
            },
            {
              status: 'insufficient_evidence',
              label: '资料不足，未生成完整回答',
              reason: '当前范围内没有找到足够可靠的片段。',
              answer_mode: 'no_answer',
            },
            {
              status: 'local_model_unavailable',
              label: '本地回答模型暂不可用',
              reason: '已找到相关资料，但这次只能显示引用片段。',
              answer_mode: 'snippet_fallback',
            },
            {
              status: 'vector_store_unavailable',
              label: '向量检索暂不可用',
              reason: '已改用关键词检索生成回答。',
              answer_mode: 'generated',
            },
          ];
          const host = document.createElement('div');
          host.innerHTML = cases.map(item => answerQualityMarkup(item)).join('');
          document.body.appendChild(host);
          const rendered = host.innerText || '';
          host.remove();
          return rendered;
        }
        """
    )
    required = (
        "已基于本地资料回答",
        "资料不足，未生成完整回答",
        "本地回答模型暂不可用",
        "当前只显示引用片段",
        "向量检索暂不可用",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise AssertionError(f"answer quality states missing: {missing}")
    return {"checked": len(required)}


def check_keyboard_focus(page: Any, timeout_ms: int) -> dict[str, str]:
    page.locator("#nav-chat").click(timeout=timeout_ms)
    page.keyboard.press("Tab")
    page.wait_for_function(
        "() => document.activeElement && document.activeElement !== document.body",
        timeout=timeout_ms,
    )
    active = page.evaluate(
        """
        () => document.activeElement.id ||
          document.activeElement.getAttribute('aria-label') ||
          document.activeElement.textContent.trim().slice(0, 32)
        """
    )
    return {"focused": active}

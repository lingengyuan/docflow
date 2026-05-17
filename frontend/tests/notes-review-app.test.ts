import { describe, expect, it } from 'vitest';

import {
  buildKnowledgeReviewViewModel,
  buildRelationshipOpportunityAction,
} from '../src/notes-review-app';

describe('notes review component model', () => {
  it('normalizes the knowledge loop into renderable state', () => {
    const model = buildKnowledgeReviewViewModel({
      signals: { files: 2, questions: 1, backlinks: 1, source_links: 1 },
      workflow: {
        completed: 6,
        total: 7,
        next_step: { detail: '标记最近回答是否有用' },
        steps: [
          { id: 'sources', title: '资料', count: 2, complete: true, detail: '已有资料' },
          { id: 'feedback', title: '反馈', count: 0, complete: false, detail: '标记反馈' },
        ],
      },
      knowledge_depth: {
        concepts: [{ title: 'DocFlow', file_count: 2, question_count: 1 }],
        source_trails: [{ question: 'DocFlow 是什么？', files: [{ id: 9 }] }],
        coverage_gaps: [{ title: '补齐来源', file: { id: 8 } }],
        next_actions: [{ title: '保存回答', file_id: 7 }],
      },
      relationship_timeline: [{ source: { id: 4 }, note: { id: 5 } }],
      review_queue: [{ file: { id: 3 }, reason: '最近引用', priority: 2 }],
      recommendations: [{ title: '继续回顾', file_id: 6 }],
      topic_activity: [{ title: 'RAG', file_count: 3 }],
    });

    expect(model.countLabel).toBe('1 项');
    expect(model.workflow.steps.map(step => step.id)).toEqual(['sources', 'feedback']);
    expect(model.workflow.completed).toBe(6);
    expect(model.concepts[0].title).toBe('DocFlow');
    expect(model.trails[0].question).toBe('DocFlow 是什么？');
    expect(model.relationships).toHaveLength(1);
    expect(model.topics[0].title).toBe('RAG');
  });

  it('builds a safe user action for saving a related-source suggestion', () => {
    const action = buildRelationshipOpportunityAction({
      source: { id: 12, file_name: 'source.md' },
      target: { id: 18, file_name: 'target.md' },
      shared_terms: ['引用', '回顾', '反馈', '额外', '忽略'],
    });

    expect(action.previewId).toBe(12);
    expect(action.sourceId).toBe(12);
    expect(action.targetId).toBe(18);
    expect(action.canSave).toBe(true);
    expect(action.terms).toEqual(['引用', '回顾', '反馈', '额外']);
  });
});

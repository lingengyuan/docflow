function renderKnowledgeReview(review) {
  window.DocFlowNotesReviewApp?.renderKnowledgeReview(review);
}

async function confirmKnowledgeRelationship(sourceId, targetId, button) {
  await window.DocFlowNotesReviewApp?.confirmKnowledgeRelationship(sourceId, targetId, button);
}

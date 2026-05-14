export type KnowledgeCardProps = {
  title: string;
  sourceCount: number;
};

export function renderKnowledgeCard(props: KnowledgeCardProps): string {
  return `${props.title}: typescript component parser evidence`;
}

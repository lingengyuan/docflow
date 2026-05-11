type SourceStatus = "supported" | "unsupported";

export function sourceLabel(status: SourceStatus): string {
  return status === "supported" ? "typescript parser evidence" : "needs review";
}

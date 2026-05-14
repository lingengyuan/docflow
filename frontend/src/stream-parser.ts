export type SseEvent = {
  event: string;
  data: string;
};

export type SseParseResult = {
  events: SseEvent[];
  buffer: string;
};

export function consumeSseBuffer(input: string): SseParseResult {
  const parts = input.split('\n\n');
  const buffer = parts.pop() ?? '';
  const events: SseEvent[] = [];

  for (const part of parts) {
    let event = '';
    const dataLines: string[] = [];

    for (const line of part.split('\n')) {
      if (line.startsWith('event: ')) {
        event = line.slice(7);
      } else if (line.startsWith('data: ')) {
        dataLines.push(line.slice(6));
      }
    }

    if (event) {
      events.push({ event, data: dataLines.join('\n') });
    }
  }

  return { events, buffer };
}


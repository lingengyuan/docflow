import { describe, expect, it } from 'vitest';

import { consumeSseBuffer } from '../src/stream-parser';

describe('consumeSseBuffer', () => {
  it('returns complete events and keeps the trailing partial buffer', () => {
    const result = consumeSseBuffer(
      'event: token\ndata: "hello"\n\n' +
        'event: done\ndata: {"history_id": 7}\n\n' +
        'event: token\ndata: "part',
    );

    expect(result.events).toEqual([
      { event: 'token', data: '"hello"' },
      { event: 'done', data: '{"history_id": 7}' },
    ]);
    expect(result.buffer).toBe('event: token\ndata: "part');
  });

  it('supports multi-line data payloads', () => {
    const result = consumeSseBuffer('event: token\ndata: first\ndata: second\n\n');

    expect(result.events).toEqual([{ event: 'token', data: 'first\nsecond' }]);
    expect(result.buffer).toBe('');
  });

  it('drops malformed records without an event type', () => {
    const result = consumeSseBuffer('data: ignored\n\n');

    expect(result.events).toEqual([]);
    expect(result.buffer).toBe('');
  });
});


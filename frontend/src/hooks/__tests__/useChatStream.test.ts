// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import { describe, it, expect, vi } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useChatStream } from '../useChatStream';
import type { ChatMessage } from '../../types/chat';

/**
 * Build a fake `Response` with a body that yields the supplied SSE chunks.
 * Each chunk is JSON-stringified and prefixed with `data: ` (server contract).
 */
const buildResponse = (chunks: object[], status = 200): Response => {
  const lines = chunks.map((c) => `data: ${JSON.stringify(c)}\n`).join('');
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(new TextEncoder().encode(lines));
      controller.close();
    },
  });
  return new Response(stream, { status });
};

/**
 * Drain processStream against a fake stream, accumulating every
 * updateMessage call so tests can inspect the final assistant payload.
 */
const drain = async (chunks: object[], status = 200) => {
  const { result } = renderHook(() => useChatStream());
  const updates: Array<Partial<ChatMessage>> = [];
  const updateMessage = vi.fn(
    (_id: string, update: Partial<ChatMessage>) => {
      updates.push({ ...update });
    }
  );
  await act(async () => {
    result.current.startStream();
    await result.current.processStream(
      buildResponse(chunks, status),
      'assistant-1',
      updateMessage
    );
  });
  return { updates, last: updates[updates.length - 1] };
};

describe('useChatStream — legacy non-agentic path (regression)', () => {
  it('concatenates delta.content when chunks have no event_type', async () => {
    const { last } = await drain([
      { choices: [{ delta: { content: 'Hello, ' } }] },
      { choices: [{ delta: { content: 'world.' } }] },
      { choices: [{ delta: {}, finish_reason: 'stop' }] },
    ]);
    expect(last.content).toBe('Hello, world.');
    expect(last.reasoning_steps).toBeUndefined();
  });

  // Regression: the live BE actually emits `event_type: null` (not omitted)
  // for every standard (non-agentic) chunk. A `??` chain in the parser
  // previously left `eventType` as `null` instead of `undefined`, which
  // routed the chunk into the forward-compat branch and silently dropped
  // `delta.content`. The bubble showed sources but no answer.
  it('treats explicit event_type: null as legacy, not as an unknown event', async () => {
    const { last } = await drain([
      {
        choices: [{ delta: { content: 'Hello, ' } }],
        event_type: null,
        stage: null,
      },
      {
        choices: [{ delta: { content: 'world.' } }],
        event_type: null,
        stage: null,
      },
      {
        choices: [{ delta: {}, finish_reason: 'stop' }],
        event_type: null,
        stage: null,
      },
    ]);
    expect(last.content).toBe('Hello, world.');
    expect(last.reasoning_steps).toBeUndefined();
  });

  it('captures standard RAG reasoning_content in the reasoning panel model', async () => {
    const { last } = await drain([
      {
        choices: [
          {
            delta: {
              content: '',
              reasoning_content: 'I need to inspect the retrieved context. ',
            },
          },
        ],
        event_type: null,
        stage: null,
      },
      {
        choices: [
          {
            delta: {
              content: 'The answer is ',
              reasoning_content: 'The context supports a direct answer.',
            },
          },
        ],
        event_type: null,
        stage: null,
      },
      {
        choices: [{ delta: { content: '42.' } }],
        event_type: null,
        stage: null,
      },
      { choices: [{ delta: {}, finish_reason: 'stop' }] },
    ]);

    expect(last.content).toBe('The answer is 42.');
    expect(last.reasoning_steps).toHaveLength(1);
    expect(last.reasoning_steps?.[0]).toMatchObject({
      stage: 'rag',
      reasoning:
        'I need to inspect the retrieved context. The context supports a direct answer.',
      status: 'done',
    });
  });

  it('extracts citations from sources/results without event_type', async () => {
    const { last } = await drain([
      {
        choices: [{ delta: { content: 'Answer' } }],
        citations: {
          results: [
            {
              content: 'snippet',
              document_name: 'a.pdf',
              document_type: 'text',
              score: 0.9,
              stage: 'rag',
            },
          ],
        },
      },
      { choices: [{ delta: {}, finish_reason: 'stop' }] },
    ]);
    expect(last.citations).toHaveLength(1);
    expect(last.citations?.[0]).toMatchObject({
      source: 'a.pdf',
      stage: 'rag',
    });
  });
});

describe('useChatStream — agentic streaming path (PR #512)', () => {
  it('routes final_answer chunks to content and other events to reasoning steps', async () => {
    const { last } = await drain([
      {
        choices: [
          {
            delta: {
              event_type: 'stage_start',
              stage: 'plan',
              reasoning_content: 'Planning the retrieval strategy…',
            },
          },
        ],
      },
      {
        choices: [
          {
            delta: {
              event_type: 'intermediate_reasoning',
              stage: 'plan',
              reasoning_content: 'Step 1: scope discovery. ',
            },
          },
        ],
      },
      {
        choices: [
          {
            delta: {
              event_type: 'intermediate_reasoning',
              stage: 'plan',
              reasoning_content: 'Step 2: focused retrieval.',
            },
          },
        ],
      },
      {
        choices: [
          {
            delta: {
              event_type: 'stage_end',
              stage: 'plan',
              reasoning_content: 'Plan ready.',
            },
          },
        ],
      },
      {
        choices: [
          {
            delta: {
              event_type: 'stage_start',
              stage: 'synthesize',
              reasoning_content: 'Composing the final answer…',
            },
          },
        ],
      },
      {
        choices: [
          {
            delta: {
              event_type: 'final_answer',
              stage: 'synthesize',
              content: 'The lion ',
            },
          },
        ],
      },
      {
        choices: [
          {
            delta: {
              event_type: 'final_answer',
              stage: 'synthesize',
              content: 'is the king.',
            },
          },
        ],
      },
      {
        choices: [
          {
            delta: {
              event_type: 'stage_end',
              stage: 'synthesize',
              reasoning_content: 'Done.',
            },
          },
        ],
      },
      { choices: [{ delta: {}, finish_reason: 'stop' }] },
    ]);

    expect(last.content).toBe('The lion is the king.');
    expect(last.reasoning_steps).toHaveLength(2);
    const [planStep, synthStep] = last.reasoning_steps!;
    expect(planStep).toMatchObject({
      stage: 'plan',
      label: 'Planning the retrieval strategy…',
      summary: 'Plan ready.',
      status: 'done',
    });
    expect(planStep.reasoning).toBe(
      'Step 1: scope discovery. Step 2: focused retrieval.'
    );
    expect(synthStep.stage).toBe('synthesize');
    expect(synthStep.status).toBe('done');
  });

  it('attaches citations + metrics when they arrive on the first final_answer chunk', async () => {
    const { last } = await drain([
      {
        choices: [
          {
            delta: {
              event_type: 'final_answer',
              stage: 'synthesize',
              content: 'ok',
            },
          },
        ],
        citations: {
          results: [
            {
              content: 'snippet',
              document_name: 'doc.pdf',
              document_type: 'text',
              score: 0.7,
              stage: 'verify_execute',
            },
          ],
        },
        metrics: {
          rag_ttft_ms: 120,
          llm_ttft_ms: 80,
          llm_generation_time_ms: 1500,
        },
      },
      { choices: [{ delta: {}, finish_reason: 'stop' }] },
    ]);

    expect(last.citations?.[0].stage).toBe('verify_execute');
    expect(last.metrics).toEqual({
      rag_ttft_ms: 120,
      llm_ttft_ms: 80,
      llm_generation_time_ms: 1500,
    });
  });

  it('lazy-creates a step when reasoning_content arrives without stage_start', async () => {
    const { last } = await drain([
      {
        choices: [
          {
            delta: {
              event_type: 'intermediate_reasoning',
              stage: 'execute',
              reasoning_content: 'orphan token',
            },
          },
        ],
      },
      { choices: [{ delta: {}, finish_reason: 'stop' }] },
    ]);
    expect(last.reasoning_steps).toHaveLength(1);
    expect(last.reasoning_steps?.[0]).toMatchObject({
      stage: 'execute',
      reasoning: 'orphan token',
      status: 'done',
    });
  });

  it('routes intermediate_output to step.output, distinct from reasoning', async () => {
    const { last } = await drain([
      {
        choices: [
          {
            delta: {
              event_type: 'stage_start',
              stage: 'execute',
              reasoning_content: 'Running tasks…',
            },
          },
        ],
      },
      {
        choices: [
          {
            delta: {
              event_type: 'intermediate_output',
              stage: 'execute',
              reasoning_content: '{"tasks":[{"id":1}]}',
            },
          },
        ],
      },
      {
        choices: [
          {
            delta: {
              event_type: 'stage_end',
              stage: 'execute',
              reasoning_content: 'Found 4 results.',
            },
          },
        ],
      },
      { choices: [{ delta: {}, finish_reason: 'stop' }] },
    ]);
    expect(last.reasoning_steps?.[0].output).toBe('{"tasks":[{"id":1}]}');
    expect(last.reasoning_steps?.[0].reasoning).toBe('');
  });

  it('marks the message as errored on event_type=error and stops accumulating answer', async () => {
    const { last } = await drain([
      {
        choices: [
          {
            delta: {
              event_type: 'stage_start',
              stage: 'plan',
              reasoning_content: 'Planning…',
            },
          },
        ],
      },
      {
        choices: [
          {
            delta: {
              event_type: 'error',
              stage: 'plan',
              content: 'pipeline failed',
              reasoning_content: 'detail: timeout',
            },
          },
        ],
      },
      { choices: [{ delta: {}, finish_reason: 'stop' }] },
    ]);
    expect(last.is_error).toBe(true);
    expect(last.content).toBe('pipeline failed');
    const step = last.reasoning_steps?.[0];
    expect(step?.status).toBe('error');
    expect(step?.output).toContain('timeout');
  });

  it('forward-compat: agent_event / unknown event types preserve reasoning_content', async () => {
    const { last } = await drain([
      {
        choices: [
          {
            delta: {
              event_type: 'agent_event',
              stage: 'execute',
              reasoning_content: 'mid-stage update',
            },
          },
        ],
      },
      { choices: [{ delta: {}, finish_reason: 'stop' }] },
    ]);
    expect(last.reasoning_steps?.[0].reasoning).toBe('mid-stage update');
  });

  it('closes any still-running step when the trailing finish_reason chunk arrives', async () => {
    const { last } = await drain([
      {
        choices: [
          {
            delta: {
              event_type: 'stage_start',
              stage: 'execute',
              reasoning_content: 'Running tasks…',
            },
          },
        ],
      },
      // No matching stage_end — server may drop one if pipeline interrupts.
      { choices: [{ delta: {}, finish_reason: 'stop' }] },
    ]);
    expect(last.reasoning_steps?.[0].status).toBe('done');
  });
});

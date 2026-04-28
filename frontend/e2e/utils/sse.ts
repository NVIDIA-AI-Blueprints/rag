/**
 * Utilities for building Server-Sent Events (SSE) style responses that the
 * RAG frontend consumes at /api/generate.
 *
 * The frontend reads raw `fetch` + `response.body.getReader()` and splits on
 * `\n`, looking for lines prefixed with `data: `. See
 * `frontend/src/api/useChatStream.ts`.
 *
 * Supported chunk shape (what the client actually reads):
 *   {
 *     choices: [{ delta: { content: string }, finish_reason: string | null }]
 *     citations?: { results: CitationSource[] }
 *     sources?: { results: CitationSource[] }
 *   }
 */

export interface CitationSource {
  content?: string;
  text?: string;
  document_name?: string;
  source?: string;
  title?: string;
  document_type?: string;
  score?: number;
  confidence_score?: number;
  similarity_score?: number;
  metadata?: Record<string, unknown>;
}

export interface StreamChunk {
  choices?: Array<{
    delta?: { content?: string; role?: string };
    message?: {
      content?: string;
      citations?: CitationSource[];
      sources?: CitationSource[];
    };
    finish_reason?: string | null;
  }>;
  citations?: { results: CitationSource[] };
  sources?: { results: CitationSource[] };
}

export interface BuildStreamOptions {
  /** Text to stream back; will be split into chunks. */
  text?: string;
  /** Number of chunks to split `text` into (default 4). */
  chunks?: number;
  /** Emit citations in the final chunk via `citations.results`. */
  citations?: CitationSource[];
  /**
   * Where to put citations in the stream.
   * - `final-top-level-citations` (default): emits `citations.results` in the last chunk
   * - `final-top-level-sources`: emits `sources.results`
   * - `final-message-citations`: emits `choices[0].message.citations`
   * - `final-message-sources`: emits `choices[0].message.sources`
   */
  citationPath?:
    | 'final-top-level-citations'
    | 'final-top-level-sources'
    | 'final-message-citations'
    | 'final-message-sources';
  /** Finish reason for last chunk (default 'stop'). */
  finishReason?: string;
  /** If true, terminate with `data: [DONE]\n\n`. Default true. */
  includeDone?: boolean;
}

const splitText = (text: string, count: number): string[] => {
  if (count <= 1 || text.length <= count) return [text];
  const size = Math.ceil(text.length / count);
  const out: string[] = [];
  for (let i = 0; i < text.length; i += size) {
    out.push(text.slice(i, i + size));
  }
  return out;
};

/**
 * Render a list of StreamChunk objects as an SSE-style string body.
 */
export function renderStreamBody(
  chunks: StreamChunk[],
  { includeDone = true }: { includeDone?: boolean } = {},
): string {
  const lines = chunks.map((c) => `data: ${JSON.stringify(c)}\n\n`);
  if (includeDone) lines.push(`data: [DONE]\n\n`);
  return lines.join('');
}

/**
 * High-level builder that takes an assistant reply string and returns an
 * SSE body that streams it in N chunks with a final `finish_reason: stop`.
 */
export function buildStreamBody(options: BuildStreamOptions = {}): string {
  const {
    text = 'Hello from mocked RAG.',
    chunks: chunkCount = 4,
    citations,
    citationPath = 'final-top-level-citations',
    finishReason = 'stop',
    includeDone = true,
  } = options;

  const pieces = splitText(text, chunkCount);
  const chunks: StreamChunk[] = pieces.map((piece, idx) => {
    const isLast = idx === pieces.length - 1;
    const chunk: StreamChunk = {
      choices: [
        {
          delta: { content: piece },
          finish_reason: isLast ? finishReason : null,
        },
      ],
    };
    if (isLast && citations && citations.length > 0) {
      if (citationPath === 'final-top-level-citations') {
        chunk.citations = { results: citations };
      } else if (citationPath === 'final-top-level-sources') {
        chunk.sources = { results: citations };
      } else if (citationPath === 'final-message-citations') {
        chunk.choices![0].message = { citations };
      } else if (citationPath === 'final-message-sources') {
        chunk.choices![0].message = { sources: citations };
      }
    }
    return chunk;
  });

  return renderStreamBody(chunks, { includeDone });
}

/**
 * Build an SSE body that simulates a backend error arriving mid-stream
 * after one or more valid chunks.
 */
export function buildPartialErrorStream(partialText: string): string {
  return renderStreamBody([
    {
      choices: [
        { delta: { content: partialText }, finish_reason: null },
      ],
    },
    // Then a malformed/error-ish chunk (still valid JSON)
    {
      choices: [
        {
          delta: {},
          finish_reason: 'error',
        },
      ],
    },
  ]);
}

export const DEFAULT_CITATIONS: CitationSource[] = [
  {
    content: 'Primary source passage about the answer.',
    document_name: 'primary-doc.pdf',
    document_type: 'text',
    score: 0.87,
  },
  {
    text: 'Secondary supporting passage.',
    source: 'secondary-doc.pdf',
    document_type: 'text',
    score: 0.72,
  },
];

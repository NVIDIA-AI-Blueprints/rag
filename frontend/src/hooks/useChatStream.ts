// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import { useState, useRef } from "react";
import type {
  AgenticMetrics,
  ChatMessage,
  Citation,
  ReasoningStep,
} from "../types/chat";

/**
 * Interface representing the state of a chat stream.
 */
export interface StreamState {
  content: string;
  citations: ChatMessage["citations"];
  error: string | null;
  isTyping: boolean;
}

/** Server-emitted `event_type` discriminators (see PR #512). */
type AgenticEventType =
  | "stage_start"
  | "stage_end"
  | "intermediate_reasoning"
  | "intermediate_output"
  | "final_reasoning"
  | "final_answer"
  | "agent_event"
  | "error";

const STAGE_START_EVENT: AgenticEventType = "stage_start";
const STAGE_END_EVENT: AgenticEventType = "stage_end";
const FINAL_ANSWER_EVENT: AgenticEventType = "final_answer";
const ERROR_EVENT: AgenticEventType = "error";
const OUTPUT_EVENT: AgenticEventType = "intermediate_output";
const STANDARD_RAG_STAGE = "rag";

/** Lazy-create or reuse the open step for an incoming chunk's stage. */
const ensureOpenStep = (
  steps: ReasoningStep[],
  stage: string | undefined
): ReasoningStep => {
  const last = steps[steps.length - 1];
  if (last && last.status === "running" && (!stage || last.stage === stage)) {
    return last;
  }
  const next: ReasoningStep = {
    stage: stage ?? "unknown",
    reasoning: "",
    output: "",
    status: "running",
  };
  steps.push(next);
  return next;
};

const closeRunningSteps = (
  steps: ReasoningStep[],
  status: "done" | "error" = "done"
) => {
  for (const step of steps) {
    if (step.status === "running") step.status = status;
  }
};

const parseSources = (raw: unknown): ChatMessage["citations"] => {
  if (!Array.isArray(raw)) return undefined;
  const scored: ChatMessage["citations"] = raw.map(
    (src: Record<string, unknown>) => {
      const score =
        typeof src.score === "string" || typeof src.score === "number"
          ? src.score
          : typeof src.confidence_score === "string" ||
              typeof src.confidence_score === "number"
            ? src.confidence_score
            : typeof src.similarity_score === "string" ||
                typeof src.similarity_score === "number"
              ? src.similarity_score
              : undefined;
      const stage =
        typeof src.stage === "string" && src.stage.length > 0
          ? src.stage
          : undefined;
      return {
        text: String(src.content || src.text || ""),
        source: String(
          src.document_name || src.source || src.title || "Unknown"
        ),
        document_type: (src.document_type as Citation["document_type"]) || "text",
        score,
        stage,
      };
    }
  );
  return scored;
};

/**
 * Custom hook for managing chat streaming functionality.
 *
 * Handles two SSE contracts on `/generate`:
 *
 * 1. **Standard / non-agentic** (and `agentic` with `enable_streaming=false`):
 *    chunks have no `event_type`; `delta.content` is concatenated into the
 *    user-facing answer; optional `delta.reasoning_content` is captured as a
 *    single standard RAG reasoning step; citations attach on the final chunk.
 *
 * 2. **Agentic streaming** (PR #512): each chunk carries an `event_type`
 *    that discriminates whether the payload is final-answer text, reasoning
 *    trace, intermediate output, or a stage boundary. The hook builds a
 *    `reasoning_steps[]` trace alongside the final answer; citations and
 *    metrics arrive on the first `final_answer` chunk.
 *
 * @returns An object containing stream state and control functions
 *
 * @example
 * ```tsx
 * const { streamState, startStream, processStream, stopStream } = useChatStream();
 * const controller = startStream();
 * await processStream(response, messageId, updateMessage);
 * ```
 */
export const useChatStream = () => {
  const [streamState, setStreamState] = useState<StreamState>({
    content: "",
    citations: [],
    error: null,
    isTyping: false,
  });

  const abortControllerRef = useRef<AbortController | null>(null);

  const startStream = () => {
    const controller = new AbortController();
    abortControllerRef.current = controller;
    setStreamState({ content: "", citations: [], error: null, isTyping: true });
    return controller;
  };

  const stopStream = () => {
    abortControllerRef.current?.abort();
    setStreamState((prev) => ({ ...prev, isTyping: false }));
  };

  const resetStream = () => {
    setStreamState({ content: "", citations: [], error: null, isTyping: false });
  };

  const processStream = async (
    response: Response,
    assistantId: string,
    updateMessage: (id: string, update: Partial<ChatMessage>) => void
  ) => {
    const reader = response.body?.getReader();
    if (!reader) throw new Error("No response body in stream");

    const decoder = new TextDecoder();
    let buffer = "";
    let content = "";
    let latestCitations: ChatMessage["citations"] = [];
    const steps: ReasoningStep[] = [];
    let metrics: AgenticMetrics | undefined;
    let isError = response.status >= 400;

    const emit = () => {
      updateMessage(assistantId, {
        content,
        citations:
          latestCitations && latestCitations.length ? latestCitations : undefined,
        is_error: isError,
        reasoning_steps: steps.length ? [...steps] : undefined,
        metrics,
      });
    };

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;

          const json = JSON.parse(line.slice(6));
          const choice = json.choices?.[0];
          const delta = choice?.delta ?? {};
          // Standard (non-agentic) responses ship `event_type: null` on
          // every chunk; we collapse that to `undefined` so the dispatch
          // below routes them through the legacy path that concatenates
          // `delta.content` into the assistant message.
          const rawEventType =
            (delta.event_type as AgenticEventType | null | undefined) ??
            (choice?.message?.event_type as
              | AgenticEventType
              | null
              | undefined) ??
            (json.event_type as AgenticEventType | null | undefined);
          const eventType: AgenticEventType | undefined =
            rawEventType ?? undefined;
          const stage: string | undefined =
            (typeof delta.stage === "string" ? delta.stage : undefined) ??
            (typeof choice?.message?.stage === "string"
              ? choice.message.stage
              : undefined) ??
            (typeof json.stage === "string" ? json.stage : undefined);

          const reasoningContent: string | undefined =
            typeof delta.reasoning_content === "string"
              ? delta.reasoning_content
              : undefined;
          const deltaContent: string | undefined =
            typeof delta.content === "string" ? delta.content : undefined;

          // Branch on event_type. When event_type is absent (standard /
          // non-agentic stream, or pre-#512 backend), fall through the
          // legacy path: append delta.content to the answer.
          if (eventType === STAGE_START_EVENT) {
            closeRunningSteps(steps);
            steps.push({
              stage: stage ?? "unknown",
              label: reasoningContent || undefined,
              reasoning: "",
              output: "",
              status: "running",
            });
          } else if (eventType === STAGE_END_EVENT) {
            const open = steps[steps.length - 1];
            if (open && open.status === "running") {
              if (reasoningContent) open.summary = reasoningContent;
              open.status = "done";
            }
          } else if (
            eventType === "intermediate_reasoning" ||
            eventType === "final_reasoning"
          ) {
            if (reasoningContent) {
              const step = ensureOpenStep(steps, stage);
              step.reasoning += reasoningContent;
            }
          } else if (eventType === OUTPUT_EVENT) {
            if (reasoningContent) {
              const step = ensureOpenStep(steps, stage);
              step.output += reasoningContent;
            }
          } else if (eventType === ERROR_EVENT) {
            isError = true;
            if (deltaContent) content += deltaContent;
            if (reasoningContent) {
              const step = ensureOpenStep(steps, stage);
              step.output += reasoningContent;
              step.status = "error";
            } else {
              closeRunningSteps(steps, "error");
            }
          } else if (eventType === FINAL_ANSWER_EVENT) {
            if (deltaContent) content += deltaContent;
          } else if (eventType !== undefined) {
            // agent_event or future / unknown event types:
            // forward-compat — capture any reasoning_content into the
            // current step so we don't lose information.
            if (reasoningContent) {
              const step = ensureOpenStep(steps, stage);
              step.reasoning += reasoningContent;
            }
          } else {
            // Legacy non-agentic chunk: no event_type discriminator.
            if (reasoningContent) {
              const step = ensureOpenStep(steps, stage ?? STANDARD_RAG_STAGE);
              step.reasoning += reasoningContent;
            }
            if (deltaContent) content += deltaContent;
          }

          // Citations and metrics may arrive on any agentic chunk, but the
          // server contract pins them to the first `final_answer` chunk.
          // We accept whichever arrives first / most recent.
          const sources =
            json.citations?.results ??
            json.sources?.results ??
            choice?.message?.citations ??
            choice?.message?.sources ??
            [];
          const parsed = parseSources(sources);
          if (parsed && parsed.length) latestCitations = parsed;

          if (json.metrics && typeof json.metrics === "object") {
            metrics = { ...(metrics ?? {}), ...(json.metrics as AgenticMetrics) };
          }

          emit();

          if (choice?.finish_reason === "stop") {
            closeRunningSteps(steps);
            emit();
            setStreamState({
              content,
              citations: latestCitations,
              error: null,
              isTyping: false,
            });
            return;
          }
        }
      }
    } catch (err: unknown) {
      if (err instanceof Error && err.name === "AbortError") return;
      setStreamState((prev) => ({
        ...prev,
        error: "Error processing stream",
        isTyping: false,
      }));
      throw err;
    } finally {
      reader.releaseLock();
    }
  };

  return {
    streamState,
    startStream,
    stopStream,
    resetStream,
    processStream,
    isStreaming: streamState.isTyping,
  };
};

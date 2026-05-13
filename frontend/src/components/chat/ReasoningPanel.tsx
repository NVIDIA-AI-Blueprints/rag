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

import { useCallback, useEffect, useState } from "react";
import { AnimatedChevron, Block, Flex, Stack, Text } from "@kui/react";
import { AlertCircle, Brain, CheckCircle2, Loader2 } from "lucide-react";
import type { ReasoningStep } from "../../types/chat";
import { useCitationUtils } from "../../hooks/useCitationUtils";

interface ReasoningPanelProps {
  /** Reasoning trace from the stream. */
  steps: ReasoningStep[];
  /** Whether the parent message is still streaming. */
  streaming?: boolean;
}

const StepIcon = ({ status }: { status: ReasoningStep["status"] }) => {
  if (status === "running") {
    return (
      <Loader2
        size={14}
        style={{
          color: "var(--text-color-subtle)",
          animation: "var(--animate-spin)",
        }}
        aria-label="Step running"
      />
    );
  }
  if (status === "error") {
    return (
      <AlertCircle
        size={14}
        style={{ color: "var(--text-color-feedback-danger)" }}
        aria-label="Step errored"
      />
    );
  }
  return (
    <CheckCircle2
      size={14}
      style={{ color: "var(--text-color-feedback-success)" }}
      aria-label="Step completed"
    />
  );
};

const StepRow = ({
  step,
  formatStage,
}: {
  step: ReasoningStep;
  formatStage: (stage: string) => string;
}) => {
  const headline = step.summary ?? step.label;
  return (
    <Stack
      gap="density-xs"
      data-testid={`reasoning-step-${step.stage}`}
      data-status={step.status}
    >
      <Flex align="center" gap="density-xs">
        <StepIcon status={step.status} />
        <Text kind="body/bold/sm">{formatStage(step.stage)}</Text>
        {headline && (
          <Text
            kind="body/regular/sm"
            style={{ color: "var(--text-color-subtle)" }}
          >
            — {headline}
          </Text>
        )}
      </Flex>
      {(step.reasoning || step.output) && (
        <Block style={{ paddingLeft: "22px" }}>
          {step.reasoning && (
            <Text
              kind="body/regular/sm"
              style={{
                color: "var(--text-color-subtle)",
                whiteSpace: "pre-wrap",
                fontFamily: "var(--font-mono)",
                fontSize: "0.85em",
              }}
            >
              {step.reasoning}
            </Text>
          )}
          {step.output && (
            <Text
              kind="body/regular/sm"
              style={{
                color: "var(--text-color-subtle)",
                whiteSpace: "pre-wrap",
                fontFamily: "var(--font-mono)",
                fontSize: "0.85em",
              }}
            >
              {step.output}
            </Text>
          )}
        </Block>
      )}
    </Stack>
  );
};

/**
 * Collapsible "Thinking" panel that renders the reasoning trace above
 * the assistant's final answer.
 *
 * - Hidden when there are no steps.
 * - Auto-expanded while the message is streaming so users can watch the
 *   agent work; collapses on completion to a "Thought for N steps" line.
 * - Step labels are formatted via `useCitationUtils.formatStage` so any
 *   future graph node renders without code changes.
 */
export const ReasoningPanel = ({ steps, streaming = false }: ReasoningPanelProps) => {
  const { formatStage } = useCitationUtils();
  const [open, setOpen] = useState<boolean>(streaming);
  const [autoCollapsed, setAutoCollapsed] = useState<boolean>(false);

  // Auto-collapse exactly once when streaming finishes.
  useEffect(() => {
    if (!streaming && !autoCollapsed) {
      setOpen(false);
      setAutoCollapsed(true);
    }
    if (streaming && autoCollapsed) {
      setAutoCollapsed(false);
    }
  }, [streaming, autoCollapsed]);

  const toggle = useCallback(() => setOpen((prev) => !prev), []);

  if (!steps || steps.length === 0) return null;

  const stepCount = steps.length;
  const headerLabel = streaming
    ? `Thinking (${stepCount} step${stepCount === 1 ? "" : "s"})`
    : `Thought for ${stepCount} step${stepCount === 1 ? "" : "s"}`;

  return (
    <Block
      data-testid="reasoning-panel"
      data-state={open ? "open" : "closed"}
      data-streaming={streaming ? "true" : "false"}
      style={{
        borderLeft: "2px solid var(--text-color-subtle)",
        paddingLeft: "8px",
        marginBottom: "8px",
      }}
    >
      <Flex
        align="center"
        gap="density-xs"
        role="button"
        tabIndex={0}
        aria-expanded={open}
        aria-label={open ? "Hide reasoning trace" : "Show reasoning trace"}
        onClick={toggle}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            toggle();
          }
        }}
        data-testid="reasoning-panel-toggle"
        style={{ cursor: "pointer", userSelect: "none" }}
      >
        <AnimatedChevron state={open ? "open" : "closed"} />
        <Brain size={14} style={{ color: "var(--text-color-subtle)" }} aria-hidden />
        <Text
          kind="body/bold/sm"
          style={{ color: "var(--text-color-subtle)" }}
        >
          {headerLabel}
        </Text>
      </Flex>
      {open && (
        <Block style={{ paddingTop: "8px" }}>
          <Stack gap="density-sm">
            {steps.map((step, index) => (
              <StepRow
                key={`${step.stage}-${index}`}
                step={step}
                formatStage={formatStage}
              />
            ))}
          </Stack>
        </Block>
      )}
    </Block>
  );
};

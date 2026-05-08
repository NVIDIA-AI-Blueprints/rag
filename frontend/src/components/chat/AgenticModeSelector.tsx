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

import { useCallback, useMemo } from "react";
import { Dropdown, Flex, Text } from "@kui/react";
import { Workflow } from "lucide-react";
import { useSettingsStore, type AgenticMode } from "../../store/useSettingsStore";

/**
 * Static metadata for each agentic mode. Kept as a top-level constant so
 * the dropdown items are referentially stable across renders.
 */
const MODE_OPTIONS: ReadonlyArray<{
  id: AgenticMode;
  label: string;
  description: string;
}> = [
  {
    id: "off",
    label: "Standard",
    description: "Standard RAG pipeline",
  },
  {
    id: "on",
    label: "Agentic",
    description: "LangGraph plan-and-execute",
  },
] as const;

const MODE_LABELS: Record<AgenticMode, string> = MODE_OPTIONS.reduce(
  (acc, opt) => {
    acc[opt.id] = opt.label;
    return acc;
  },
  {} as Record<AgenticMode, string>
);

/**
 * Two-state selector that controls the per-request `agentic` flag on the
 * `/generate` endpoint.
 *
 * - `Standard` (default): sends `agentic: false` to force the standard
 *   RAG pipeline.
 * - `Agentic`: sends `agentic: true` to use the LangGraph plan-and-execute
 *   pipeline.
 *
 * (The previous `Auto` mode — omit the field, let the server decide — was
 * dropped per the #514 review: with only two real outcomes a third option
 * was just noise.)
 *
 * The selection is persisted via `useSettingsStore`, matching how every
 * other chat setting is stored.
 *
 * @returns A Dropdown-backed pill that shows the current pipeline mode.
 */
export const AgenticModeSelector = () => {
  const agenticMode = useSettingsStore((s) => s.agenticMode);
  const setSettings = useSettingsStore((s) => s.set);

  const handleSelect = useCallback(
    (mode: AgenticMode) => {
      setSettings({ agenticMode: mode });
    },
    [setSettings]
  );

  const items = useMemo(
    () =>
      MODE_OPTIONS.map((opt) => ({
        children: (
          <Flex
            direction="col"
            gap="density-xs"
            data-testid={`agentic-mode-option-${opt.id}`}
            data-active={opt.id === agenticMode ? "true" : "false"}
          >
            <Text kind="body/bold/md">{opt.label}</Text>
            <Text
              kind="body/regular/sm"
              style={{ color: "var(--text-color-subtle)" }}
            >
              {opt.description}
            </Text>
          </Flex>
        ),
        onSelect: () => handleSelect(opt.id),
      })),
    [agenticMode, handleSelect]
  );

  return (
    <Flex align="center" gap="density-sm" data-testid="agentic-mode-selector">
      <Flex align="center" gap="density-xs">
        <Workflow
          size={14}
          style={{ color: "var(--text-color-subtle)" }}
          aria-hidden
        />
        <Text
          kind="label/regular/lg"
          style={{ color: "var(--text-color-subtle)" }}
        >
          Pipeline:
        </Text>
      </Flex>
      <Dropdown
        items={items}
        size="small"
        side="top"
        align="start"
        aria-label="Select RAG pipeline mode"
        data-testid="agentic-mode-dropdown"
      >
        <span
          role="button"
          tabIndex={0}
          aria-label={`Pipeline mode: ${MODE_LABELS[agenticMode]}`}
          data-testid="agentic-mode-trigger"
          data-mode={agenticMode}
          style={{ cursor: "pointer", display: "inline-flex" }}
        >
          <Text kind="body/bold/sm">{MODE_LABELS[agenticMode]}</Text>
        </span>
      </Dropdown>
    </Flex>
  );
};

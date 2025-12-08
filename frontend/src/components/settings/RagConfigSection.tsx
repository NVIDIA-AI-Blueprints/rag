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

import { Stack } from "@kui/react";
import { useSettingsStore, useServerDefaultsStore } from "../../store/useSettingsStore";
import { SettingSlider, SettingInput } from "./SettingControls";
import { useCallback } from "react";

/**
 * RAG Configuration section component for adjusting retrieval and generation settings.
 * 
 * All values start as undefined in the store. Display values are shown from server defaults
 * for accurate user experience. Nothing is persisted until user explicitly interacts with controls.
 * 
 * @returns RAG configuration section with sliders and inputs
 */
export const RagConfigSection = () => {
  const { 
    temperature, 
    topP, 
    confidenceScoreThreshold,
    vdbTopK,
    rerankerTopK,
    maxTokens,
    set: setSettings 
  } = useSettingsStore();

  // Get server defaults for accurate display values
  const { config: serverDefaults } = useServerDefaultsStore();
  const defaults = serverDefaults?.rag_configuration;

  // Simple handlers - directly set values when user interacts
  const handleVdbTopKChange = useCallback((value: string) => {
    const num = parseInt(value, 10);
    if (!isNaN(num) && num > 0) {
      setSettings({ vdbTopK: num });
    } else if (value === "") {
      setSettings({ vdbTopK: undefined });
    }
  }, [setSettings]);

  const handleRerankerTopKChange = useCallback((value: string) => {
    const num = parseInt(value, 10);
    if (!isNaN(num) && num > 0) {
      setSettings({ rerankerTopK: num });
    } else if (value === "") {
      setSettings({ rerankerTopK: undefined });
    }
  }, [setSettings]);

  const handleMaxTokensChange = useCallback((value: string) => {
    const num = parseInt(value, 10);
    if (!isNaN(num) && num > 0) {
      setSettings({ maxTokens: num });
    } else if (value === "") {
      setSettings({ maxTokens: undefined });
    }
  }, [setSettings]);

  return (
    <Stack gap="6">
      <SettingSlider
        label="Temperature"
        description={`Controls randomness in responses. Higher = more creative, lower = more focused. Default: ${defaults?.temperature ?? 'N/A'}`}
        value={temperature ?? defaults?.temperature ?? 0}
        onChange={(value) => setSettings({ temperature: value })}
        min={0}
        max={1}
        step={0.1}
        data-testid="temperature-slider"
      />

      <SettingSlider
        label="Top P"
        description={`Limits token selection to cumulative probability. Lower = more focused. Default: ${defaults?.top_p ?? 'N/A'}`}
        value={topP ?? defaults?.top_p ?? 1.0}
        onChange={(value) => setSettings({ topP: value })}
        min={0}
        max={1}
        step={0.1}
        data-testid="top-p-slider"
      />

      <SettingSlider
        label="Confidence Score Threshold"
        description={`Minimum confidence for document relevance. Higher = more selective. Default: ${defaults?.confidence_threshold ?? 'N/A'}`}
        value={confidenceScoreThreshold ?? defaults?.confidence_threshold ?? 0.0}
        onChange={(value) => setSettings({ confidenceScoreThreshold: value })}
        min={0}
        max={1}
        step={0.05}
        data-testid="confidence-threshold-slider"
      />

      <SettingInput
        label="Vector DB Top K"
        description={`Number of documents to retrieve from vector database. Default: ${defaults?.vdb_top_k ?? 'N/A'}`}
        value={(vdbTopK ?? defaults?.vdb_top_k ?? "").toString()}
        onChange={handleVdbTopKChange}
        type="number"
        min={1}
        max={400}
      />

      <SettingInput
        label="Reranker Top K"
        description={`Number of documents to return after reranking. Default: ${defaults?.reranker_top_k ?? 'N/A'}`}
        value={(rerankerTopK ?? defaults?.reranker_top_k ?? "").toString()}
        onChange={handleRerankerTopKChange}
        type="number"
        min={1}
        max={50}
      />

      <SettingInput
        label="Max Tokens"
        description={`Maximum number of tokens in the response. Default: ${defaults?.max_tokens ?? 'N/A'}`}
        value={(maxTokens ?? defaults?.max_tokens ?? "").toString()}
        onChange={handleMaxTokensChange}
        type="number"
        min={1}
        max={128000}
      />
    </Stack>
  );
};
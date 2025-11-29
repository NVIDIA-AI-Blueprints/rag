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

import { Stack, FormField, TextInput } from "@kui/react";
import { useSettingsStore, useServerDefaultsStore } from "../../store/useSettingsStore";

/**
 * Endpoints section component for configuring API endpoint URLs.
 * 
 * Provides input fields for configuring LLM, embedding, reranker, VLM,
 * and vector database endpoints. Shows current values as placeholders
 * and allows override with custom URLs.
 * 
 * @returns Endpoints configuration section with URL input fields
 */
export const EndpointsSection = () => {
  const { 
    llmEndpoint, 
    embeddingEndpoint, 
    rerankerEndpoint, 
    vlmEndpoint, 
    vdbEndpoint, 
    set: setSettings 
  } = useSettingsStore();

  // Get server defaults for accurate display
  const { config: serverDefaults } = useServerDefaultsStore();
  const defaults = serverDefaults?.endpoints;

  const endpoints = [
    { key: 'llmEndpoint', label: 'LLM Endpoint', value: llmEndpoint, defaultValue: defaults?.llm_endpoint },
    { key: 'embeddingEndpoint', label: 'Embedding Endpoint', value: embeddingEndpoint, defaultValue: defaults?.embedding_endpoint },
    { key: 'rerankerEndpoint', label: 'Reranker Endpoint', value: rerankerEndpoint, defaultValue: defaults?.reranker_endpoint },
    { key: 'vlmEndpoint', label: 'VLM Endpoint', value: vlmEndpoint, defaultValue: defaults?.vlm_endpoint },
    { key: 'vdbEndpoint', label: 'Vector Database Endpoint', value: vdbEndpoint, defaultValue: defaults?.vdb_endpoint },
  ];

  return (
    <Stack gap="4" slotDivider={<hr />}>
      {endpoints.map(({ key, label, value, defaultValue }) => (
        <FormField
          key={key}
          slotLabel={label}
          slotHelp={defaultValue ? `Default: ${defaultValue}` : undefined}
        >
          {(args) => (
            <TextInput
              {...args}
              value={value ?? defaultValue ?? ""}
              onValueChange={(newValue) => setSettings({ [key]: newValue.trim() === "" ? undefined : newValue })}
            />
          )}
        </FormField>
      ))}
    </Stack>
  );
}; 
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

import { Stack, FormField, TextInput, Flex, Spinner } from "@kui/react";
import { useSettingsStore, useHealthDependentFeatures } from "../../store/useSettingsStore";

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
  const { isHealthLoading, shouldDisableHealthFeatures } = useHealthDependentFeatures();

  const endpoints = [
    { key: 'llmEndpoint', label: 'LLM Endpoint', value: llmEndpoint },
    { key: 'embeddingEndpoint', label: 'Embedding Endpoint', value: embeddingEndpoint },
    { key: 'rerankerEndpoint', label: 'Reranker Endpoint', value: rerankerEndpoint },
    { key: 'vlmEndpoint', label: 'VLM Endpoint', value: vlmEndpoint },
    { key: 'vdbEndpoint', label: 'Vector Database Endpoint', value: vdbEndpoint },
  ];

  return (
    <Stack gap="4" slotDivider={<hr />}>
      {endpoints.map(({ key, label, value }) => (
        <FormField
          key={key}
          slotLabel={
            <Flex align="center" gap="density-sm">
              {label}
              {isHealthLoading && <Spinner size="small" aria-label="Loading endpoint configuration" />}
            </Flex>
          }
          slotHelp={
            isHealthLoading 
              ? "Loading endpoint from system configuration..." 
              : shouldDisableHealthFeatures
                ? "System configuration unavailable"
                : "Leave empty to use default endpoint"
          }
        >
          {(args) => (
            <TextInput
              {...args}
              value={value ?? ""}
              onValueChange={(newValue) => setSettings({ [key]: newValue.trim() === "" ? undefined : newValue })}
              placeholder={
                isHealthLoading
                  ? "Loading from system configuration..."
                  : shouldDisableHealthFeatures
                    ? "System configuration unavailable"
                    : value 
                      ? `Current: ${value}` 
                      : "Leave empty for default"
              }
              disabled={shouldDisableHealthFeatures}
            />
          )}
        </FormField>
      ))}
    </Stack>
  );
}; 
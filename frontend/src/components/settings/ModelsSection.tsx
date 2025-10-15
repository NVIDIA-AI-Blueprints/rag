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
 * Models section component for configuring AI model settings.
 * 
 * Uses KUI FormField and TextInput components for consistent form styling.
 * Provides input fields for configuring LLM, embedding, reranker, and VLM models.
 * 
 * @returns Models configuration section with KUI form components
 */
export const ModelsSection = () => {
  const { model, embeddingModel, rerankerModel, vlmModel, set: setSettings } = useSettingsStore();
  const { isHealthLoading, shouldDisableHealthFeatures } = useHealthDependentFeatures();

  const models = [
    { key: 'model', label: 'LLM Model', value: model },
    { key: 'embeddingModel', label: 'Embedding Model', value: embeddingModel },
    { key: 'rerankerModel', label: 'Reranker Model', value: rerankerModel },
    { key: 'vlmModel', label: 'VLM Model', value: vlmModel },
  ];

  return (
    <Stack gap="4" slotDivider={<hr />}>
      {models.map(({ key, label, value }) => (
        <FormField
          key={key}
          slotLabel={
            <Flex align="center" gap="density-sm">
              {label}
              {isHealthLoading && <Spinner size="small" aria-label="Loading model configuration" />}
            </Flex>
          }
          slotHelp={
            isHealthLoading 
              ? "Loading model from system configuration..." 
              : shouldDisableHealthFeatures
                ? "System configuration unavailable"
                : "Leave empty to use default model"
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
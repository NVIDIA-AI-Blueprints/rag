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

import { useSettingsStore } from "../../store/useSettingsStore";
import { CollapsibleSection } from "../ui/CollapsibleSection";
import { SettingTextInput } from "./SettingControls";
import { Flex, Text, Card, FormField, TextInput } from "@kui/react";
import { ChevronDown, ChevronRight } from "lucide-react";

interface ModelConfigSectionProps {
  isExpanded: boolean;
  onToggle: () => void;
}

interface EndpointConfigSectionProps {
  isExpanded: boolean;
  onToggle: () => void;
}

interface AdvancedSettingsSectionProps {
  stopTokensInput: string;
  onStopTokensChange: (value: string) => void;
}

interface SettingsSectionProps {
  title: string;
  isExpanded: boolean;
  onToggle: () => void;
  icon: React.ReactNode;
  children: React.ReactNode;
}

export const SettingsSection = ({ title, isExpanded, onToggle, icon, children }: SettingsSectionProps) => (
  <>
    <Card
      onClick={onToggle}
      style={{ cursor: 'pointer', marginBottom: 'var(--spacing-density-md)' }}
      data-testid="settings-section-button"
    >
      <Flex justify="between" align="center">
        <Flex align="center" gap="density-md">
          <div 
            style={{ color: 'var(--color-brand)' }}
            data-testid="settings-section-icon"
          >
            {icon}
          </div>
          <Text 
            kind="label/bold/lg"
            data-testid="settings-section-title"
          >
            {title}
          </Text>
        </Flex>
        <div 
          style={{ color: 'var(--text-color-subtle)' }}
          data-testid="settings-section-chevron"
        >
          {isExpanded ? (
            <ChevronDown size={20} data-testid="chevron-down" />
          ) : (
            <ChevronRight size={20} data-testid="chevron-right" />
          )}
        </div>
      </Flex>
    </Card>
    
    <div 
      className={`${isExpanded ? "block" : "hidden"} space-y-4 ml-4 mb-8`}
      data-testid="settings-section-content"
    >
      {children}
    </div>
  </>
);


export const ModelConfigSection = ({ isExpanded, onToggle }: ModelConfigSectionProps) => {
  const {
    model,
    embeddingModel,
    rerankerModel,
    vlmModel,
    set: setSettings,
  } = useSettingsStore();

  const models = [
    { key: 'model', label: 'LLM Model', value: model ?? "" },
    { key: 'embeddingModel', label: 'Embedding Model', value: embeddingModel ?? "" },
    { key: 'rerankerModel', label: 'Reranker Model', value: rerankerModel ?? "" },
    { key: 'vlmModel', label: 'VLM Model', value: vlmModel ?? "" },
  ];

  return (
    <CollapsibleSection title="Model Configuration" isExpanded={isExpanded} onToggle={onToggle}>
      {models.map(({ key, label, value }) => (
        <SettingTextInput
          key={key}
          label={label}
          value={value}
          onChange={(newValue) => setSettings({ [key]: newValue })}
        />
      ))}
    </CollapsibleSection>
  );
};

export const EndpointConfigSection = ({ isExpanded, onToggle }: EndpointConfigSectionProps) => {
  const {
    llmEndpoint,
    embeddingEndpoint,
    rerankerEndpoint,
    vlmEndpoint,
    vdbEndpoint,
    set: setSettings,
  } = useSettingsStore();

  const endpoints = [
    { key: 'llmEndpoint', label: 'LLM Endpoint', value: llmEndpoint ?? "" },
    { key: 'embeddingEndpoint', label: 'Embedding Endpoint', value: embeddingEndpoint ?? "" },
    { key: 'rerankerEndpoint', label: 'Reranker Endpoint', value: rerankerEndpoint ?? "" },
    { key: 'vlmEndpoint', label: 'VLM Endpoint', value: vlmEndpoint ?? "" },
    { key: 'vdbEndpoint', label: 'Vector Database Endpoint', value: vdbEndpoint ?? "" },
  ];

  return (
    <CollapsibleSection title="Endpoint Configuration" isExpanded={isExpanded} onToggle={onToggle}>
      {endpoints.map(({ key, label, value }) => (
        <SettingTextInput
          key={key}
          label={label}
          value={value}
          onChange={(newValue) => setSettings({ [key]: newValue })}
          placeholder="Leave empty for default"
        />
      ))}
    </CollapsibleSection>
  );
};

export const AdvancedSettingsSection = ({ 
  stopTokensInput, 
  onStopTokensChange 
}: AdvancedSettingsSectionProps) => (
  <div 
    className="mb-6"
    data-testid="advanced-settings-section"
  >
    <Text 
      kind="label/bold/sm"
      style={{ marginBottom: 'var(--spacing-density-sm)', display: 'block' }}
      data-testid="advanced-settings-title"
    >
      Other Settings
    </Text>
    <Card>
      <FormField
        slotLabel="Stop Tokens"
        slotHelp="Tokens that will stop text generation when encountered."
      >
        <TextInput
          value={stopTokensInput}
          onValueChange={onStopTokensChange}
          placeholder="Enter tokens separated by commas"
          data-testid="stop-tokens-input"
        />
      </FormField>
    </Card>
  </div>
);

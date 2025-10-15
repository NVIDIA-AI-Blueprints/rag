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
    <button
      onClick={onToggle}
      className="mb-4 flex w-full items-center justify-between p-4 rounded-xl bg-neutral-900/50 border border-neutral-700 hover:border-neutral-600 transition-all duration-200"
      data-testid="settings-section-button"
    >
      <div className="flex items-center gap-3">
        <div 
          className="text-[var(--nv-green)]"
          data-testid="settings-section-icon"
        >
          {icon}
        </div>
        <span 
          className="text-lg font-medium text-white"
          data-testid="settings-section-title"
        >
          {title}
        </span>
      </div>
      <div 
        className="text-gray-400"
        data-testid="settings-section-chevron"
      >
        {isExpanded ? (
          <svg 
            className="w-5 h-5" 
            fill="none" 
            stroke="currentColor" 
            strokeWidth="2" 
            viewBox="0 0 24 24"
            data-testid="chevron-down"
          >
            <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
          </svg>
        ) : (
          <svg 
            className="w-5 h-5" 
            fill="none" 
            stroke="currentColor" 
            strokeWidth="2" 
            viewBox="0 0 24 24"
            data-testid="chevron-right"
          >
            <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
          </svg>
        )}
      </div>
    </button>
    
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
    <h3 
      className="text-sm font-medium mb-3"
      data-testid="advanced-settings-title"
    >
      Advanced Settings
    </h3>
    <div className="p-3 rounded-lg bg-neutral-900/50 border border-neutral-700">
      <label 
        className="block text-sm font-medium mb-2"
        data-testid="stop-tokens-label"
      >
        Stop Tokens
      </label>
      <input
        type="text"
        value={stopTokensInput}
        onChange={(e) => onStopTokensChange(e.target.value)}
        className="w-full rounded-lg bg-neutral-800 border border-neutral-600 px-3 py-2 text-sm text-white focus:ring-2 focus:ring-[var(--nv-green)]/50"
        placeholder="Enter tokens separated by commas"
        data-testid="stop-tokens-input"
      />
      <p 
        className="mt-2 text-xs text-gray-400"
        data-testid="stop-tokens-description"
      >
        Tokens that will stop text generation when encountered.
      </p>
    </div>
  </div>
); 
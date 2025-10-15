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

import { Stack, Switch, Text } from "@kui/react";
import { useSettingsStore } from "../../store/useSettingsStore";

/**
 * Props for the FeatureToggle component.
 */
interface FeatureToggleProps {
  slotLabel: string;
  description: string;
  value: boolean;
  featureKey: string;
  onShowWarning: (key: string, value: boolean) => void;
}

const FeatureToggle = ({ slotLabel, description, value, featureKey, onShowWarning }: FeatureToggleProps) => (
  <Stack gap="1">
    <Switch
      slotLabel={slotLabel}
      checked={value}
      onCheckedChange={(checked) => onShowWarning(featureKey, checked)}
    />
    <Text kind="body/regular/sm" className="text-subtle">
      {description}
    </Text>
  </Stack>
);

/**
 * Props for the FeatureTogglesSection component.
 */
interface FeatureTogglesSectionProps {
  onShowWarning: (key: string, value: boolean) => void;
}

/**
 * Feature toggles section component for enabling/disabling experimental features.
 * 
 * Provides toggle controls for reranker, citations, guardrails, query rewriting,
 * VLM inference, and filter generation features.
 * 
 * @param props - Component props with warning handler
 * @returns Feature toggles section with toggle switches
 */
export const FeatureTogglesSection = ({ onShowWarning }: FeatureTogglesSectionProps) => {
  const {
    enableReranker,
    includeCitations,
    useGuardrails,
    enableQueryRewriting,
    enableVlmInference,
    enableFilterGenerator,
  } = useSettingsStore();

  const features = [
    { key: 'enableReranker', label: 'Enable Reranker', desc: 'Use reranking to improve document relevance', value: enableReranker ?? true },
    { key: 'includeCitations', label: 'Include Citations', desc: 'Add source citations to responses', value: includeCitations ?? true },
    { key: 'useGuardrails', label: 'Use Guardrails', desc: 'Apply safety guardrails to responses', value: useGuardrails ?? false },
    { key: 'enableQueryRewriting', label: 'Query Rewriting', desc: 'Rewrite user queries for better retrieval', value: enableQueryRewriting ?? false },
    { key: 'enableVlmInference', label: 'VLM Inference', desc: 'Enable vision-language model inference', value: enableVlmInference ?? false },
    { key: 'enableFilterGenerator', label: 'Filter Generator', desc: 'Auto-generate filters from queries', value: enableFilterGenerator ?? false },
  ];

  return (
    <Stack gap="4" slotDivider={<hr />}>
      {features.map(({ key, label, desc, value }) => (
        <FeatureToggle
          key={key}
          featureKey={key}
          slotLabel={label}
          description={desc}
          value={value}
          onShowWarning={onShowWarning}
        />
      ))}
    </Stack>
  );
}; 
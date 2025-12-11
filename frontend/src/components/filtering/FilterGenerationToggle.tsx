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

import React, { useState, useCallback } from "react";
import { Switch, Text, Flex, Stack, Card, Button, FormField, TextInput, Slider, Tooltip } from "@kui/react";
import { Info, Settings, X } from "lucide-react";
import type { FilterGenerationConfig } from "../../types/collections";

interface FilterGenerationToggleProps {
  enabled: boolean;
  config?: FilterGenerationConfig;
  onToggle: (enabled: boolean) => void;
  onConfigChange?: (config: FilterGenerationConfig) => void;
  className?: string;
}

export const FilterGenerationToggle: React.FC<FilterGenerationToggleProps> = ({
  enabled,
  config,
  onToggle,
  onConfigChange,
  className = ""
}) => {
  const [showConfig, setShowConfig] = useState(false);
  const [tempConfig, setTempConfig] = useState<FilterGenerationConfig>(
    config || {
      enable_filter_generator: enabled,
      model_name: "nvidia/llama-3.3-nemotron-super-49b-v1",
      temperature: 0.1,
      top_p: 0.9,
      max_tokens: 500
    }
  );

  const handleToggle = useCallback(() => {
    const newEnabled = !enabled;
    onToggle(newEnabled);
    
    if (onConfigChange) {
      onConfigChange({
        ...tempConfig,
        enable_filter_generator: newEnabled
      });
    }
  }, [enabled, onToggle, onConfigChange, tempConfig]);

  const handleConfigSave = useCallback(() => {
    if (onConfigChange) {
      onConfigChange({
        ...tempConfig,
        enable_filter_generator: enabled
      });
    }
    setShowConfig(false);
  }, [onConfigChange, tempConfig, enabled]);

  const handleConfigChange = useCallback((field: keyof FilterGenerationConfig, value: string | number | boolean) => {
    setTempConfig(prev => ({
      ...prev,
      [field]: value
    }));
  }, []);

  return (
    <Stack gap="density-md" className={className}>
      {/* Toggle Switch */}
      <Flex justify="between" align="center">
        <Flex align="center" gap="density-md">
          <Flex align="center" gap="density-sm">
            <Switch
              checked={enabled}
              onCheckedChange={handleToggle}
            />
            <Text kind="label/bold/sm">
              Natural Language Filter Generation
            </Text>
          </Flex>
          
          <Tooltip content="Automatically converts your natural language queries into precise metadata filters using LLMs.">
            <Info size={16} style={{ color: 'var(--text-color-subtle)' }} />
          </Tooltip>
        </Flex>

        {/* Configuration Button */}
        {onConfigChange && (
          <Button
            onClick={() => setShowConfig(!showConfig)}
            kind="tertiary"
            size="small"
            disabled={!enabled}
            title="Configure filter generation settings"
          >
            <Settings size={16} />
          </Button>
        )}
      </Flex>

      {/* Status Indicator */}
      <Flex align="center" gap="density-sm">
        <div style={{ 
          width: '8px', 
          height: '8px', 
          borderRadius: '50%',
          backgroundColor: enabled ? 'var(--feedback-color-success)' : 'var(--text-color-subtle)'
        }} />
        <Text kind="body/regular/xs" style={{ color: 'var(--text-color-subtle)' }}>
          {enabled ? "AI filter generation enabled" : "Using manual filters only"}
        </Text>
      </Flex>

      {/* Configuration Panel */}
      {showConfig && onConfigChange && (
        <Card>
          <Flex justify="between" align="center" style={{ marginBottom: 'var(--spacing-density-md)' }}>
            <Text kind="label/bold/sm">Filter Generation Configuration</Text>
            <Button
              onClick={() => setShowConfig(false)}
              kind="tertiary"
              size="small"
            >
              <X size={16} />
            </Button>
          </Flex>

          <Stack gap="density-md">
            <FormField slotLabel="Model Name">
              <TextInput
                value={tempConfig.model_name || ""}
                onValueChange={(value) => handleConfigChange("model_name", value)}
                placeholder="nvidia/llama-3.3-nemotron-super-49b-v1"
              />
            </FormField>

            <FormField slotLabel="Server URL (optional)">
              <TextInput
                value={tempConfig.server_url || ""}
                onValueChange={(value) => handleConfigChange("server_url", value)}
                placeholder="Leave empty for default endpoint"
              />
            </FormField>

            <FormField slotLabel={`Temperature (${tempConfig.temperature})`}>
              <Slider
                min={0}
                max={1}
                step={0.1}
                value={[tempConfig.temperature || 0.1]}
                onValueChange={([value]) => handleConfigChange("temperature", value)}
              />
            </FormField>

            <FormField slotLabel="Max Tokens">
              <TextInput
                type="number"
                value={String(tempConfig.max_tokens || 500)}
                onValueChange={(value) => handleConfigChange("max_tokens", parseInt(value))}
              />
            </FormField>

            <Flex justify="end" gap="density-sm">
              <Button
                onClick={() => setShowConfig(false)}
                kind="tertiary"
                size="small"
              >
                Cancel
              </Button>
              <Button
                onClick={handleConfigSave}
                kind="primary"
                color="brand"
                size="small"
              >
                Save
              </Button>
            </Flex>
          </Stack>
        </Card>
      )}
    </Stack>
  );
};

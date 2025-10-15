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
import type { FilterGenerationConfig } from "../../types/collections";

interface FilterGenerationToggleProps {
  enabled: boolean;
  config?: FilterGenerationConfig;
  onToggle: (enabled: boolean) => void;
  onConfigChange?: (config: FilterGenerationConfig) => void;
  className?: string;
}

const InfoIcon = () => (
  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
  </svg>
);

const SettingsIcon = () => (
  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
  </svg>
);

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
    <div className={`space-y-3 ${className}`}>
      {/* Toggle Switch */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={enabled}
              onChange={handleToggle}
              className="rounded border-neutral-600 bg-neutral-700 text-[var(--nv-green)] focus:ring-1 focus:ring-[var(--nv-green)]"
            />
            <span className="text-sm font-medium text-gray-300">
              Natural Language Filter Generation
            </span>
          </label>
          
          <div className="group relative">
            <InfoIcon />
            <div className="invisible group-hover:visible absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 w-80 p-3 bg-neutral-800 border border-neutral-600 rounded-lg text-xs text-gray-300 z-10">
              <div className="font-medium mb-1">AI-Powered Filtering</div>
              <p>Automatically converts your natural language queries into precise metadata filters using LLMs. 
                 For example: &quot;Show me AI documents with high ratings&quot; becomes &quot;category == &apos;AI&apos; AND rating &gt; 4.0&quot;</p>
              <div className="mt-2 text-neutral-400">
                <strong>Requirements:</strong> Milvus vector database, compatible LLM endpoint
              </div>
            </div>
          </div>
        </div>

        {/* Configuration Button */}
        {onConfigChange && (
          <button
            type="button"
            onClick={() => setShowConfig(!showConfig)}
            disabled={!enabled}
            className="p-1.5 text-neutral-400 hover:text-[var(--nv-green)] disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            title="Configure filter generation settings"
          >
            <SettingsIcon />
          </button>
        )}
      </div>

      {/* Status Indicator */}
      <div className="flex items-center gap-2 text-xs">
        <div className={`w-2 h-2 rounded-full ${enabled ? 'bg-green-400' : 'bg-neutral-500'}`} />
        <span className="text-neutral-400">
          {enabled ? "AI filter generation enabled" : "Using manual filters only"}
        </span>
      </div>

      {/* Configuration Panel */}
      {showConfig && onConfigChange && (
        <div className="p-4 bg-neutral-800/50 rounded-lg border border-neutral-600 space-y-4">
          <div className="flex items-center justify-between">
            <h4 className="text-sm font-medium text-white">Filter Generation Configuration</h4>
            <button
              onClick={() => setShowConfig(false)}
              className="text-neutral-400 hover:text-white"
            >
              &times;
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Model Name */}
            <div>
              <label className="block text-xs font-medium text-gray-300 mb-1">
                Model Name
              </label>
              <input
                type="text"
                value={tempConfig.model_name || ""}
                onChange={(e) => handleConfigChange("model_name", e.target.value)}
                placeholder="nvidia/llama-3.3-nemotron-super-49b-v1"
                className="w-full px-2 py-1 rounded bg-neutral-700 border border-neutral-600 text-white text-xs focus:outline-none focus:ring-1 focus:ring-[var(--nv-green)]"
              />
            </div>

            {/* Server URL */}
            <div>
              <label className="block text-xs font-medium text-gray-300 mb-1">
                Server URL (optional)
              </label>
              <input
                type="text"
                value={tempConfig.server_url || ""}
                onChange={(e) => handleConfigChange("server_url", e.target.value)}
                placeholder="Leave empty for default endpoint"
                className="w-full px-2 py-1 rounded bg-neutral-700 border border-neutral-600 text-white text-xs focus:outline-none focus:ring-1 focus:ring-[var(--nv-green)]"
              />
            </div>

            {/* Temperature */}
            <div>
              <label className="block text-xs font-medium text-gray-300 mb-1">
                Temperature ({tempConfig.temperature})
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={tempConfig.temperature || 0.1}
                onChange={(e) => handleConfigChange("temperature", parseFloat(e.target.value))}
                className="w-full"
              />
            </div>

            {/* Max Tokens */}
            <div>
              <label className="block text-xs font-medium text-gray-300 mb-1">
                Max Tokens
              </label>
              <input
                type="number"
                min="100"
                max="2000"
                value={tempConfig.max_tokens || 500}
                onChange={(e) => handleConfigChange("max_tokens", parseInt(e.target.value))}
                className="w-full px-2 py-1 rounded bg-neutral-700 border border-neutral-600 text-white text-xs focus:outline-none focus:ring-1 focus:ring-[var(--nv-green)]"
              />
            </div>
          </div>

          <div className="flex justify-end gap-2">
            <button
              onClick={() => setShowConfig(false)}
              className="px-3 py-1 text-xs text-neutral-400 hover:text-white"
            >
              Cancel
            </button>
            <button
              onClick={handleConfigSave}
              className="px-3 py-1 text-xs bg-[var(--nv-green)] text-black font-medium rounded hover:bg-[var(--nv-green)]/80"
            >
              Save
            </button>
          </div>
        </div>
      )}
    </div>
  );
}; 
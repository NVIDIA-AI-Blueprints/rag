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

import { useState, useEffect, useCallback } from "react";
import { useSettingsStore } from "../store/useSettingsStore";

/**
 * Custom hook for managing settings input handlers with validation.
 * 
 * Provides controlled input states and validation handlers for numeric settings
 * like VDB top-k, reranker top-k, max tokens, and stop tokens.
 * 
 * @returns Object with input values and change handlers for settings
 * 
 * @example
 * ```tsx
 * const { vdbTopKInput, handleVdbTopKChange, maxTokensInput, handleMaxTokensChange } = useSettingsHandlers();
 * ```
 */
export const useSettingsHandlers = () => {
  const { vdbTopK, rerankerTopK, maxTokens, stopTokens, set: setSettings } = useSettingsStore();
  
  // Provide defaults for undefined values
  const [vdbTopKInput, setVdbTopKInput] = useState((vdbTopK ?? 20).toString());
  const [rerankerTopKInput, setRerankerTopKInput] = useState((rerankerTopK ?? 5).toString());
  const [maxTokensInput, setMaxTokensInput] = useState((maxTokens ?? 1024).toString());
  const [stopTokensInput, setStopTokensInput] = useState((stopTokens ?? []).join(", "));

  useEffect(() => {
    setVdbTopKInput((vdbTopK ?? 20).toString());
    setRerankerTopKInput((rerankerTopK ?? 5).toString());
    setMaxTokensInput((maxTokens ?? 1024).toString());

  }, [vdbTopK, rerankerTopK, maxTokens]);

  const handleVdbTopKChange = useCallback((value: string) => {
    setVdbTopKInput(value);
    const num = parseInt(value, 10);
    if (!isNaN(num) && num > 0) {
      setSettings({ vdbTopK: num });
    }
  }, [setSettings]);

  const handleRerankerTopKChange = useCallback((value: string) => {
    setRerankerTopKInput(value);
    const num = parseInt(value, 10);
    if (!isNaN(num) && num > 0) {
      setSettings({ rerankerTopK: num });
    }
  }, [setSettings]);

  const handleMaxTokensChange = useCallback((value: string) => {
    setMaxTokensInput(value);
    const num = parseInt(value, 10);
    if (!isNaN(num) && num > 0) {
      setSettings({ maxTokens: num });
    }
  }, [setSettings]);

  const handleStopTokensChange = useCallback((value: string) => {
    setStopTokensInput(value);
    const tokens = value.split(",").map(t => t.trim()).filter(t => t.length > 0);
    setSettings({ stopTokens: tokens });
  }, [setSettings]);

  return {
    vdbTopKInput,
    rerankerTopKInput,
    maxTokensInput,
    stopTokensInput,
    handleVdbTopKChange,
    handleRerankerTopKChange,
    handleMaxTokensChange,
    handleStopTokensChange
  };
}; 
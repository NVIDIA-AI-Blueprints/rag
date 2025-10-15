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

import { useCallback } from "react";

/**
 * Configuration options for textarea auto-resize behavior.
 */
interface TextareaResizeOptions {
  minHeight?: number;
  maxHeight?: number;
}

/**
 * Custom hook for implementing auto-resize functionality on textarea elements.
 * 
 * Provides handlers and styles to make textareas automatically adjust their height
 * based on content, with configurable minimum and maximum height constraints.
 * 
 * @param options - Configuration options for resize behavior
 * @returns Object with input handler and style getter functions
 * 
 * @example
 * ```tsx
 * const { handleInput, getTextareaStyle } = useTextareaResize({ maxHeight: 200 });
 * <textarea onInput={handleInput} style={getTextareaStyle()} />
 * ```
 */
export const useTextareaResize = ({ 
  minHeight = 44, 
  maxHeight = 160 
}: TextareaResizeOptions = {}) => {
  
  const handleInput = useCallback((e: React.FormEvent<HTMLTextAreaElement>) => {
    const target = e.target as HTMLTextAreaElement;
    target.style.height = 'auto';
    target.style.height = Math.min(target.scrollHeight, maxHeight) + 'px';
  }, [maxHeight]);

  const getTextareaStyle = useCallback(() => ({
    minHeight: `${minHeight}px`,
    maxHeight: `${maxHeight}px`,
    height: 'auto'
  }), [minHeight, maxHeight]);

  return {
    handleInput,
    getTextareaStyle,
  };
}; 
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
 * Custom hook for generating UUIDs with fallback support for older browsers.
 * 
 * This hook provides a reliable UUID generation method that tries to use
 * crypto.randomUUID when available and falls back to a local implementation
 * for better browser compatibility.
 * 
 * @returns Object with generateUUID and generateShortUUID functions
 * 
 * @example
 * ```tsx
 * function ChatComponent() {
 *   const { generateUUID } = useUUID();
 *   
 *   const createMessage = () => {
 *     const id = generateUUID();
 *     // Use the generated UUID...
 *   };
 * }
 * ```
 */
export const useUUID = () => {
  /**
   * Generates a UUID v4 using a local implementation for better browser compatibility.
   * 
   * @returns A UUID v4 string in the format: xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx
   */
  const generateUUID = useCallback((): string => {
    // Try to use crypto.randomUUID if available (modern browsers)
    if (typeof crypto !== 'undefined' && crypto.randomUUID) {
      return crypto.randomUUID();
    }

    // Fallback implementation for older browsers or environments
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
      const r = Math.random() * 16 | 0;
      const v = c === 'x' ? r : (r & 0x3 | 0x8);
      return v.toString(16);
    });
  }, []);

  /**
   * Generates a short UUID (8 characters) for cases where a full UUID is not needed.
   * 
   * @returns A short random string of 8 hexadecimal characters
   */
  const generateShortUUID = useCallback((): string => {
    return Math.random().toString(16).substring(2, 10);
  }, []);

  return {
    generateUUID,
    generateShortUUID,
  };
}; 
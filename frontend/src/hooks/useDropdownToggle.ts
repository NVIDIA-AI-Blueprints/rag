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

import { useState, useEffect, useRef, useCallback } from "react";

/**
 * Custom hook for managing dropdown state with click-outside-to-close functionality.
 * 
 * Provides toggle, open, close actions and a ref for the dropdown container.
 * Automatically closes dropdown when clicking outside the referenced element.
 * 
 * @returns Object containing dropdown state and control functions
 * 
 * @example
 * ```tsx
 * const { isOpen, ref, toggle, close } = useDropdownToggle();
 * <div ref={ref}>
 *   <button onClick={toggle}>Toggle</button>
 *   {isOpen && <div>Dropdown content</div>}
 * </div>
 * ```
 */
export const useDropdownToggle = () => {
  const [isOpen, setIsOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  const toggle = useCallback(() => {
    setIsOpen(prev => !prev);
  }, []);

  const close = useCallback(() => {
    setIsOpen(false);
  }, []);

  const open = useCallback(() => {
    setIsOpen(true);
  }, []);

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (!ref.current?.contains(e.target as Node)) {
        setIsOpen(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  return {
    isOpen,
    ref,
    toggle,
    close,
    open,
  };
}; 
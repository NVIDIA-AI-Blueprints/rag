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

import { useCallback } from 'react';
import { useSettingsStore } from '../store/useSettingsStore';

/**
 * Custom hook for managing application theme.
 * 
 * Provides theme state and toggle functionality, persisted via settings store.
 * Automatically updates KUI ThemeProvider and applies theme classes to document.
 * 
 * @returns Object with current theme and toggle function
 * 
 * @example
 * ```tsx
 * const { theme, toggleTheme } = useTheme();
 * <ThemeProvider theme={theme}>
 *   <Button onClick={toggleTheme}>Toggle Theme</Button>
 * </ThemeProvider>
 * ```
 */
export const useTheme = () => {
  const { theme, set } = useSettingsStore();

  const toggleTheme = useCallback(() => {
    const newTheme = theme === 'dark' ? 'light' : 'dark';
    set({ theme: newTheme });
  }, [theme, set]);

  const setTheme = useCallback((newTheme: 'light' | 'dark') => {
    set({ theme: newTheme });
  }, [set]);

  return {
    theme,
    toggleTheme,
    setTheme,
    isDark: theme === 'dark',
    isLight: theme === 'light'
  };
};

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

import { useState, useCallback } from "react";
import { useSettingsStore } from "../store/useSettingsStore";

/**
 * Custom hook for managing feature warning dialogs and user consent.
 * 
 * Provides functionality to show warning modals when users enable experimental
 * or potentially impactful features. Manages user preferences for "don't show again"
 * and handles confirmation/cancellation of feature changes.
 * 
 * @returns Object with modal state and warning management functions
 * 
 * @example
 * ```tsx
 * const { showModal, showWarning, confirmChange, cancelChange } = useFeatureWarning();
 * showWarning("enableReranker", true);
 * ```
 */
export const useFeatureWarning = () => {
  const [showModal, setShowModal] = useState(false);
  const [pendingChange, setPendingChange] = useState<{ key: string; value: boolean } | null>(null);
  const [dontShowAgain, setDontShowAgain] = useState(false);
  const { set: setSettings } = useSettingsStore();

  const showWarning = useCallback((key: string, value: boolean) => {
    console.log('🔥 FEATURE WARNING TRIGGERED:', key);
    
    // If enabling and user hasn't disabled warnings, show modal
    if (value && !dontShowAgain) {
      setPendingChange({ key, value });
      setShowModal(true);
    } else {
      // If disabling or user disabled warnings, apply change directly
      setSettings({ [key]: value });
    }
  }, [dontShowAgain, setSettings]);

  const confirmChange = useCallback((shouldNotShowAgain: boolean) => {
    if (pendingChange) {
      setSettings({ [pendingChange.key]: pendingChange.value });
      setPendingChange(null);
    }
    if (shouldNotShowAgain) {
      setDontShowAgain(true);
    }
    setShowModal(false);
  }, [pendingChange, setSettings]);

  const cancelChange = useCallback(() => {
    setPendingChange(null);
    setShowModal(false);
  }, []);

  return {
    showModal,
    showWarning,
    confirmChange,
    cancelChange
  };
}; 
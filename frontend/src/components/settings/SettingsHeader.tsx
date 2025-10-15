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

/**
 * Header component for the settings page displaying title and description.
 * 
 * Shows a settings icon, title, and description text with consistent styling.
 * Used at the top of the settings page to provide context and branding.
 * 
 * @returns Settings header with icon, title, and description
 */
export const SettingsHeader = () => (
  <div 
    className="flex items-center gap-4 pb-6 border-b border-neutral-800"
    data-testid="settings-header"
  >
    <div 
      className="p-3 rounded-xl bg-[var(--nv-green)]/15 border border-[var(--nv-green)]/25"
      data-testid="settings-icon-container"
    >
      <svg 
        className="w-6 h-6 text-[var(--nv-green)]" 
        fill="none" 
        stroke="currentColor" 
        strokeWidth="2" 
        viewBox="0 0 24 24"
        data-testid="settings-icon"
      >
        <path strokeLinecap="round" strokeLinejoin="round" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
        <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
      </svg>
    </div>
    <div data-testid="settings-content">
      <h1 
        className="text-2xl font-bold text-white mb-1"
        data-testid="settings-title"
      >
        Settings
      </h1>
      <p 
        className="text-gray-400 text-sm"
        data-testid="settings-description"
      >
        Configure your RAG system parameters and preferences
      </p>
    </div>
  </div>
); 
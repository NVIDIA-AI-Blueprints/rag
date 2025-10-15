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

import React from 'react';

interface SettingsSidebarProps {
  activeSection: string;
  onSectionChange: (section: string) => void;
}

interface SettingsNavItem {
  id: string;
  title: string;
  icon: React.ReactNode;
}

const settingsNavItems: SettingsNavItem[] = [
  {
    id: 'ragConfig',
    title: 'RAG Configuration',
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
      </svg>
    ),
  },
  {
    id: 'features',
    title: 'Feature Toggles',
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 100 4m0-4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 100 4m0-4v2m0-6V4" />
      </svg>
    ),
  },
  {
    id: 'models',
    title: 'Model Configuration',
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
      </svg>
    ),
  },
  {
    id: 'endpoints',
    title: 'Endpoint Configuration',
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9v-9m0-9v9m0 9c-5 0-9-4-9-9s4-9 9-9" />
      </svg>
    ),
  },
  {
    id: 'advanced',
    title: 'Advanced Settings',
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 100 4m0-4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 100 4m0-4v2m0-6V4" />
      </svg>
    ),
  },
];

export const SettingsSidebar: React.FC<SettingsSidebarProps> = ({
  activeSection,
  onSectionChange,
}) => {
  return (
    <div className="w-72 bg-neutral-900/30 border-r border-neutral-800 px-6 py-8">
      <div className="space-y-3">
        <div className="mb-6">
          <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">
            Configuration
          </h3>
        </div>
        
        {settingsNavItems.map((item) => {
          const isActive = activeSection === item.id;
          
          return (
            <button
              key={item.id}
              onClick={() => onSectionChange(item.id)}
              className={`w-full flex items-center gap-3 px-4 py-3.5 rounded-xl text-left transition-all duration-200 group border ${
                isActive
                  ? 'bg-[var(--nv-green)]/15 border-[var(--nv-green)]/25 text-white shadow-sm'
                  : 'text-gray-300 hover:text-white hover:bg-neutral-800/40 border-transparent hover:border-neutral-700/50'
              }`}
              data-testid={`settings-nav-${item.id}`}
            >
              <div 
                className={`flex-shrink-0 transition-colors duration-200 ${
                  isActive 
                    ? 'text-[var(--nv-green)]' 
                    : 'text-gray-500 group-hover:text-gray-400'
                }`}
              >
                {item.icon}
              </div>
              <span className={`font-medium text-sm ${
                isActive ? 'text-white' : 'text-gray-300 group-hover:text-white'
              }`}>
                {item.title}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}; 
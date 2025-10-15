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

interface CollapsibleSectionProps {
  title: string;
  isExpanded: boolean;
  onToggle: () => void;
  children: React.ReactNode;
}

const ExpandIcon = ({ isExpanded }: { isExpanded: boolean }) => (
  <span className="text-sm" data-testid="expand-icon">
    {isExpanded ? "▼" : "▶"}
  </span>
);

const SectionHeader = ({ title, isExpanded, onToggle }: Pick<CollapsibleSectionProps, 'title' | 'isExpanded' | 'onToggle'>) => (
  <button
    onClick={onToggle}
    className="mb-4 flex w-full items-center justify-between text-sm font-medium hover:text-gray-300 transition-colors"
    data-testid="section-header"
    aria-expanded={isExpanded}
  >
    <span>{title}</span>
    <ExpandIcon isExpanded={isExpanded} />
  </button>
);

const SectionContent = ({ isExpanded, children }: { isExpanded: boolean; children: React.ReactNode }) => (
  <div 
    className={`${isExpanded ? "block" : "hidden"} space-y-4`}
    data-testid="section-content"
    aria-hidden={!isExpanded}
  >
    {children}
  </div>
);

export const CollapsibleSection = ({ title, isExpanded, onToggle, children }: CollapsibleSectionProps) => (
  <div className="mb-6" data-testid="collapsible-section">
    <SectionHeader title={title} isExpanded={isExpanded} onToggle={onToggle} />
    <SectionContent isExpanded={isExpanded}>
      {children}
    </SectionContent>
  </div>
); 
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

import { Flex, Text } from "@kui/react";
import { ChevronDown, ChevronRight } from "lucide-react";

interface CollapsibleSectionProps {
  title: string;
  isExpanded: boolean;
  onToggle: () => void;
  children: React.ReactNode;
}

const SectionHeader = ({ title, isExpanded, onToggle }: Pick<CollapsibleSectionProps, 'title' | 'isExpanded' | 'onToggle'>) => (
  <button
    onClick={onToggle}
    style={{ 
      width: '100%', 
      marginBottom: 'var(--spacing-density-md)',
      background: 'none',
      border: 'none',
      cursor: 'pointer',
      padding: 0
    }}
    data-testid="section-header"
    aria-expanded={isExpanded}
  >
    <Flex justify="between" align="center">
      <Text kind="label/bold/sm">{title}</Text>
      {isExpanded ? (
        <ChevronDown size={16} style={{ color: 'var(--text-color-subtle)' }} data-testid="expand-icon" />
      ) : (
        <ChevronRight size={16} style={{ color: 'var(--text-color-subtle)' }} data-testid="expand-icon" />
      )}
    </Flex>
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
  <div style={{ marginBottom: 'var(--spacing-density-lg)' }} data-testid="collapsible-section">
    <SectionHeader title={title} isExpanded={isExpanded} onToggle={onToggle} />
    <SectionContent isExpanded={isExpanded}>
      {children}
    </SectionContent>
  </div>
);

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

interface SectionConfig {
  [key: string]: boolean;
}

export const useCollapsibleSections = (initialState: SectionConfig = {}) => {
  const [expandedSections, setExpandedSections] = useState<SectionConfig>(initialState);

  const toggleSection = useCallback((key: string) => {
    setExpandedSections(prev => ({ 
      ...prev, 
      [key]: !prev[key] 
    }));
  }, []);

  const expandSection = useCallback((key: string) => {
    setExpandedSections(prev => ({ 
      ...prev, 
      [key]: true 
    }));
  }, []);

  const collapseSection = useCallback((key: string) => {
    setExpandedSections(prev => ({ 
      ...prev, 
      [key]: false 
    }));
  }, []);

  const expandAll = useCallback(() => {
    setExpandedSections(prev => {
      const newState = { ...prev };
      Object.keys(newState).forEach(key => {
        newState[key] = true;
      });
      return newState;
    });
  }, []);

  const collapseAll = useCallback(() => {
    setExpandedSections(prev => {
      const newState = { ...prev };
      Object.keys(newState).forEach(key => {
        newState[key] = false;
      });
      return newState;
    });
  }, []);

  const isExpanded = useCallback((key: string) => {
    return expandedSections[key] ?? false;
  }, [expandedSections]);

  return {
    expandedSections,
    toggleSection,
    expandSection,
    collapseSection,
    expandAll,
    collapseAll,
    isExpanded,
  };
}; 
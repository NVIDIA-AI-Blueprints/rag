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

import { ChevronRight } from "lucide-react";

interface ExpandChevronProps {
  isExpanded: boolean;
}

export const ExpandChevron = ({ isExpanded }: ExpandChevronProps) => (
  <div style={{ 
    padding: 'var(--spacing-density-xs)', 
    borderRadius: 'var(--border-radius-md)',
    transition: 'background-color 0.2s'
  }}>
    <ChevronRight 
      size={16}
      style={{ 
        color: 'var(--text-color-subtle)',
        transform: isExpanded ? 'rotate(90deg)' : 'rotate(0deg)',
        transition: 'transform 0.2s'
      }}
    />
  </div>
);

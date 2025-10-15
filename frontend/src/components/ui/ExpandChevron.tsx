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

interface ExpandChevronProps {
  isExpanded: boolean;
}

export const ExpandChevron = ({ isExpanded }: ExpandChevronProps) => (
  <div className="p-1 rounded-md hover:bg-neutral-800 transition-colors">
    <svg 
      className={`w-4 h-4 text-gray-400 group-hover:text-[var(--nv-green)] transition-all duration-200 ${
        isExpanded ? 'rotate-90' : ''
      }`} 
      fill="none" 
      stroke="currentColor" 
      strokeWidth="2" 
      viewBox="0 0 24 24"
    >
      <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
    </svg>
  </div>
); 
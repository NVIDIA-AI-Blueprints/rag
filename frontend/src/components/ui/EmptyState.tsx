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
 * Props for the EmptyState component.
 */
interface EmptyStateProps {
  title: string;
  description?: string;
  icon?: React.ReactNode;
}

/**
 * Default icon component for empty states.
 */
const DefaultEmptyIcon = () => (
  <svg className="w-10 h-10 text-gray-400" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 0 0-3.375-3.375h-1.5A1.125 1.125 0 0 1 13.5 7.125v-1.5a3.375 3.375 0 0 0-3.375-3.375H8.25m2.25 0H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 0 0-9-9Z" />
  </svg>
);

/**
 * Empty state component for displaying placeholder content when no data is available.
 * 
 * Shows a centered layout with an icon, title, and optional description.
 * Used throughout the application to provide user-friendly feedback when
 * lists or collections are empty.
 * 
 * @param props - Component props with title, description, and optional icon
 * @returns Empty state component with icon, title, and description
 */
export const EmptyState = ({ 
  title, 
  description, 
  icon = <DefaultEmptyIcon /> 
}: EmptyStateProps) => (
  <div className="flex h-[300px] flex-col items-center justify-center text-center px-4">
    <div className="w-20 h-20 rounded-full bg-neutral-800/50 border border-neutral-700 flex items-center justify-center mb-6 shadow-lg">
      {icon}
    </div>
    <h3 className="mb-3 text-xl font-medium text-gray-200">{title}</h3>
    {description && (
      <p className="text-sm text-gray-400 max-w-md leading-relaxed">{description}</p>
    )}
  </div>
); 
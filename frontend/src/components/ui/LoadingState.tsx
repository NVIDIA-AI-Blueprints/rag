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
 * Props for the LoadingState component.
 */
interface LoadingStateProps {
  message?: string;
}

/**
 * Loading state component that displays a spinner with optional message.
 * 
 * Shows a centered spinning indicator with NVIDIA green accent color
 * and an optional loading message. Used throughout the application
 * to indicate loading states.
 * 
 * @param props - Component props with optional message
 * @returns Loading state component with spinner and message
 */
export const LoadingState = ({ message = "Loading..." }: LoadingStateProps) => (
  <div className="flex h-[300px] items-center justify-center">
    <div className="flex flex-col items-center gap-4">
      <div className="h-8 w-8 animate-spin rounded-full border-3 border-[var(--nv-green)] border-t-transparent" />
      <p className="text-base text-gray-300">{message}</p>
    </div>
  </div>
); 
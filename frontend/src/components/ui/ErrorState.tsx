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
 * Props for the ErrorState component.
 */
interface ErrorStateProps {
  message?: string;
  onRetry?: () => void;
}

/**
 * Error icon component for error states.
 */
const ErrorIcon = () => (
  <svg className="w-6 h-6 text-red-400" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
  </svg>
);

/**
 * Error state component for displaying error messages with optional retry functionality.
 * 
 * Shows a centered layout with an error icon, message, and optional retry button.
 * Used throughout the application to provide user-friendly error feedback
 * and recovery options.
 * 
 * @param props - Component props with message and optional retry handler
 * @returns Error state component with icon, message, and optional retry button
 */
export const ErrorState = ({ 
  message = "Something went wrong", 
  onRetry 
}: ErrorStateProps) => (
  <div className="flex h-[300px] flex-col items-center justify-center text-center">
    <div className="w-12 h-12 rounded-full bg-red-500/20 flex items-center justify-center mb-4">
      <ErrorIcon />
    </div>
    <p className="mb-4 text-base text-red-400">{message}</p>
    {onRetry && (
      <button
        onClick={onRetry}
        className="px-4 py-2 text-sm text-[var(--nv-green)] hover:bg-[var(--nv-green)]/10 rounded-lg transition-colors"
      >
        Try again
      </button>
    )}
  </div>
); 
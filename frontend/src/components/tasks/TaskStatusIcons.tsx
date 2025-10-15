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

import type { IngestionTask } from "../../types/api";
import { useTaskUtils } from "../../hooks/useTaskUtils";

const SpinnerIcon = () => (
  <div 
    className="w-4 h-4 animate-spin rounded-full border-2 border-brand border-t-transparent" 
    data-testid="spinner-icon"
  />
);

const SuccessIcon = () => (
  <div 
    className="w-4 h-4 rounded-full bg-feedback-success flex items-center justify-center"
    data-testid="success-icon"
  >
    <svg className="w-2.5 h-2.5 text-feedback-success-inverse" fill="none" stroke="currentColor" strokeWidth="3" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
    </svg>
  </div>
);

const WarningIcon = () => (
  <div 
    className="w-4 h-4 rounded-full bg-feedback-warning flex items-center justify-center"
    data-testid="warning-icon"
  >
    <svg className="w-2.5 h-2.5 text-feedback-warning-inverse" fill="none" stroke="currentColor" strokeWidth="3" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-2.694-.833-3.464 0L3.34 16.5c-.77.833.192 2.5 1.732 2.5z" />
    </svg>
  </div>
);

const ErrorIcon = () => (
  <div 
    className="w-4 h-4 rounded-full bg-feedback-danger flex items-center justify-center"
    data-testid="error-icon"
  >
    <svg className="w-2.5 h-2.5 text-feedback-danger-inverse" fill="none" stroke="currentColor" strokeWidth="3" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
    </svg>
  </div>
);

interface TaskStatusIconProps {
  state: string;
  task?: IngestionTask & { completedAt?: number; read?: boolean };
}

export const TaskStatusIcon = ({ state, task }: TaskStatusIconProps) => {
  const { getTaskStatus } = useTaskUtils();
  
  // If we have the full task, use the enhanced status logic
  if (task) {
    // Check for PENDING first
    if (state === "PENDING") {
      return <SpinnerIcon />;
    }
    
    const status = getTaskStatus(task);
    
    if (status.isPartial) {
      return <WarningIcon />;
    } else if (status.isSuccess) {
      return <SuccessIcon />;
    } else if (status.isFailed) {
      return <ErrorIcon />;
    }
  }

  // Fallback to simple state-based logic
  switch (state) {
    case "PENDING":
      return <SpinnerIcon />;
    case "FINISHED":
      return <SuccessIcon />;
    default:
      return <ErrorIcon />;
  }
}; 
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

import { StatusMessage, Button, Flex } from "@kui/react";
import { AlertCircle } from "lucide-react";

/**
 * Props for the ErrorState component.
 */
interface ErrorStateProps {
  message?: string;
  onRetry?: () => void;
}

/**
 * Error state component for displaying error messages with optional retry functionality.
 */
export const ErrorState = ({ 
  message = "Something went wrong", 
  onRetry 
}: ErrorStateProps) => (
  <Flex 
    direction="col" 
    align="center" 
    justify="center" 
    style={{ height: '300px', textAlign: 'center' }}
  >
    <StatusMessage
      slotHeading={message}
      slotMedia={<AlertCircle size={24} style={{ color: 'var(--feedback-color-danger)' }} />}
    />
    {onRetry && (
      <Button
        onClick={onRetry}
        kind="tertiary"
        color="brand"
        size="small"
        style={{ marginTop: 'var(--spacing-density-md)' }}
      >
        Try again
      </Button>
    )}
  </Flex>
);

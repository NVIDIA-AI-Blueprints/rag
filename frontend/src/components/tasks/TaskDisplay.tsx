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

import { useCallback } from "react";
import { Card, Button, Text, Stack, ProgressBar, StatusMessage, Notification } from "@kui/react";
import type { IngestionTask } from "../../types/api";
import { TaskStatusIcon } from "./TaskStatusIcons";
import { useTaskUtils } from "../../hooks/useTaskUtils";

interface TaskDisplayProps {
  task: IngestionTask & { completedAt?: number; read?: boolean };
  onMarkRead?: () => void;
  onRemove?: () => void;
}

const RemoveIcon = () => (
  <svg 
    className="w-4 h-4" 
    fill="none" 
    stroke="currentColor" 
    strokeWidth="2" 
    viewBox="0 0 24 24"
    data-testid="remove-icon"
  >
    <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
  </svg>
);

interface TaskHeaderProps {
  task: IngestionTask & { completedAt?: number; read?: boolean };
}

const TaskHeader = ({ task }: TaskHeaderProps) => {
  const { getTaskStatus } = useTaskUtils();

  const status = getTaskStatus(task);

  return (
    <Stack gap="2" data-testid="task-header">
      <div className="flex items-start gap-3">
        <div className="flex items-center mt-0.5">
          <TaskStatusIcon state={task.state} task={task} />
        </div>
        <Stack gap="1">
          <Text 
            kind="body/semibold/md"
            className={task.read ? 'text-neutral-400' : 'text-white'}
            data-testid="task-collection-name"
          >
            {task.collection_name}
          </Text>
          <Text 
            kind="body/regular/xs"
            className={status.color}
            data-testid="task-status-text"
          >
            {status.text}
          </Text>
        </Stack>
      </div>
    </Stack>
  );
};

const TaskProgress = ({ task }: { task: TaskDisplayProps['task'] }) => {
  const { formatTimestamp } = useTaskUtils();
  const { documents = [], total_documents = 0 } = task.result || {};
  const progress = total_documents > 0 ? (documents.length / total_documents) * 100 : 0;

  return (
    <Stack gap="2" data-testid="task-progress">
      <>
          <Text 
            kind="body/regular/sm"
            className={task.read ? 'text-neutral-400' : 'text-white'}
            data-testid="progress-text"
          >
          Uploaded: {documents.length} / {total_documents}
        </Text>
        {task.completedAt && (
          <Text 
            kind="body/regular/xs"
            className="text-neutral-300"
            data-testid="completion-time"
          >
            {formatTimestamp(task.completedAt)}
          </Text>
        )}
      </>

      <ProgressBar
        value={progress}
        aria-label="Upload progress"
        data-testid="progress-bar"
      />
    </Stack>
  );
};

const TaskErrors = ({ task }: { task: TaskDisplayProps['task'] }) => {
  const { shouldHideTaskMessage } = useTaskUtils();
  const { message = "", failed_documents = [], validation_errors = [] } = task.result || {};
  const shouldHideMessage = shouldHideTaskMessage(task);

  return (
    <Stack gap="2">
      {message && !shouldHideMessage && (
        <StatusMessage
          slotHeading=""
          slotSubheading={
            <Text kind="mono/sm" className="text-red-100 whitespace-pre-wrap break-words">
              {message}
            </Text>
          }
        />
      )}

      {failed_documents.length > 0 && (
        <Notification
          status="error"
          slotHeading="Failed Documents"
          slotSubheading={`${failed_documents.length} document${failed_documents.length > 1 ? 's' : ''} failed to process`}
          slotFooter={
            <Stack gap="2" className="max-h-32 overflow-y-auto">
              {failed_documents.map((doc, i) => (
                <div key={i}>
                  <Text kind="body/semibold/sm" className="text-red-200 mb-2">
                    {doc.document_name}
                  </Text>
                  <Text kind="mono/sm" className="text-red-100 whitespace-pre-wrap break-words">
                    {doc.error_message}
                  </Text>
                </div>
              ))}
            </Stack>
          }
        />
      )}

      {validation_errors.length > 0 && (
        <Notification
          status="warning"
          slotHeading="Validation Errors"
          slotSubheading={`${validation_errors.length} validation error${validation_errors.length > 1 ? 's' : ''} found`}
        />
      )}
    </Stack>
  );
};

export const TaskDisplay = ({ task, onMarkRead, onRemove }: TaskDisplayProps) => {
  const handleMarkAsRead = useCallback(() => {
    // Only mark as read if it's unread and onMarkRead is available
    if (onMarkRead && !task.read) {
      onMarkRead();
    }
  }, [onMarkRead, task.read]);

  const clickableProps = onMarkRead && !task.read
    ? {
        tabIndex: 0,
        onClick: handleMarkAsRead,
        onFocus: handleMarkAsRead,
        onKeyDown: (e: React.KeyboardEvent) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            handleMarkAsRead();
          }
        },
      }
    : {};

      return (
      <div data-testid="task-display" {...clickableProps}>
      <Card
        interactive={onMarkRead && !task.read}
        kind="solid"
        style={{ 
          background: 'var(--border-color-interaction-inverse-pressed)',
          border: !task.read ? '1px solid var(--nv-green)' : undefined
        }}
        className={`group relative transition-all duration-200 ${onMarkRead && !task.read ? 'cursor-pointer' : ''}`}
      >
        {/* Remove button - only shows if onRemove is provided */}
        {onRemove && (
          <Button
            kind="tertiary"
            size="small"
            color="danger"
            onClick={(e) => {
              e.stopPropagation();
              onRemove();
            }}
            title="Remove notification"
            className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity z-10"
            data-testid="remove-button"
          >
            <RemoveIcon />
          </Button>
        )}

        <Stack gap="3">
          <TaskHeader task={task} />
          
          <Stack gap="2" data-testid="task-content">
            <TaskProgress task={task} />
            <TaskErrors task={task} />
          </Stack>
        </Stack>
      </Card>
    </div>
  );
}; 
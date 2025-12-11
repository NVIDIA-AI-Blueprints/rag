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
import { Card, Button, Text, Stack, ProgressBar, StatusMessage, Notification, Flex } from "@kui/react";
import { X } from "lucide-react";
import type { IngestionTask } from "../../types/api";
import { TaskStatusIcon } from "./TaskStatusIcons";
import { useTaskUtils } from "../../hooks/useTaskUtils";

interface TaskDisplayProps {
  task: IngestionTask & { completedAt?: number; read?: boolean };
  onMarkRead?: () => void;
  onRemove?: () => void;
}

interface TaskHeaderProps {
  task: IngestionTask & { completedAt?: number; read?: boolean };
}

const TaskHeader = ({ task }: TaskHeaderProps) => {
  const { getTaskStatus } = useTaskUtils();
  const status = getTaskStatus(task);

  return (
    <Stack gap="2" data-testid="task-header">
      <Flex gap="density-md" align="start">
        <div style={{ marginTop: '2px' }}>
          <TaskStatusIcon state={task.state} task={task} />
        </div>
        <Stack gap="1">
          <Text 
            kind="body/semibold/md"
            data-testid="task-collection-name"
          >
            {task.collection_name}
          </Text>
          <Text 
            kind="body/regular/xs"
            data-testid="task-status-text"
          >
            {status.text}
          </Text>
        </Stack>
      </Flex>
    </Stack>
  );
};

const TaskProgress = ({ task }: { task: TaskDisplayProps['task'] }) => {
  const { formatTimestamp } = useTaskUtils();
  const { documents = [], total_documents = 0 } = task.result || {};
  const progress = total_documents > 0 ? (documents.length / total_documents) * 100 : 0;

  return (
    <Stack gap="2" data-testid="task-progress">
      <Text 
        kind="body/regular/sm"
        data-testid="progress-text"
      >
        Uploaded: {documents.length} / {total_documents}
      </Text>
      {task.completedAt && (
        <Text 
          kind="body/regular/xs"
          data-testid="completion-time"
        >
          {formatTimestamp(task.completedAt)}
        </Text>
      )}

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
            <Text kind="mono/sm">
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
            <Stack gap="2" style={{ maxHeight: '128px', overflowY: 'auto' }}>
              {failed_documents.map((doc, i) => (
                <div key={i}>
                  <Text kind="body/semibold/sm">
                    {doc.document_name}
                  </Text>
                  <Text kind="mono/sm">
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
        selected={!task.read}
        className={onMarkRead && !task.read ? 'cursor-pointer' : ''}
      >
        <Stack gap="3">
          <Flex justify="between" align="start">
            <TaskHeader task={task} />
            
            {/* Remove button */}
            {onRemove && (
              <Button
                kind="tertiary"
                size="tiny"
                onClick={(e) => {
                  e.stopPropagation();
                  onRemove();
                }}
                title="Remove notification"
                data-testid="remove-button"
              >
                <X size={16} />
              </Button>
            )}
          </Flex>
          
          <Stack gap="2" data-testid="task-content">
            <TaskProgress task={task} />
            <TaskErrors task={task} />
          </Stack>
        </Stack>
      </Card>
    </div>
  );
};

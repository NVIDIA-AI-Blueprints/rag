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
import { useCollectionDrawerStore } from "../../store/useCollectionDrawerStore";
import { useNotificationStore } from "../../store/useNotificationStore";
import { openNotificationPanel } from "../notifications/NotificationBell";
import { Button, Flex, Stack, Text } from "@kui/react";
import type { Collection } from "../../types/collections";

/**
 * Props for the CollectionItem component.
 */
interface CollectionItemProps {
  collection: Collection;
}

const SpinnerIcon = () => (
  <div className="w-4 h-4 animate-spin rounded-full border-2 border-gray-600 border-t-[var(--nv-green)]" />
);

const MoreIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
    <circle cx="12" cy="12" r="2" />
    <circle cx="12" cy="5" r="2" />
    <circle cx="12" cy="19" r="2" />
  </svg>
);

export const CollectionItem = ({ collection }: CollectionItemProps) => {
  const { openDrawer } = useCollectionDrawerStore();
  const { getPendingTasks } = useNotificationStore();

  const pendingTasks = getPendingTasks();
  const pendingTask = pendingTasks.find(
    (t) => t.collection_name === collection.collection_name && t.state === "PENDING"
  );
  const hasPendingTasks = !!pendingTask;
  
  // Get progress for display: "3/10"
  const completedCount = pendingTask?.result?.documents_completed ?? pendingTask?.result?.documents?.length ?? 0;
  const totalCount = pendingTask?.result?.total_documents ?? 0;

  const handleOpenDrawer = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    openDrawer(collection);
  }, [collection, openDrawer]);

  return (
    <Flex justify="between" align="center">
      <Stack>
        <Text kind="body/regular/md">{collection.collection_name}</Text>
        <Text kind="body/regular/sm">
          {collection.num_entities.toLocaleString()} entities
        </Text>
      </Stack>
      {hasPendingTasks ? (
        <button
          onClick={openNotificationPanel}
          className="flex items-center gap-2 cursor-pointer hover:opacity-80 transition-opacity bg-transparent border-none p-0"
          title="View upload progress"
        >
          <Text kind="body/regular/sm" className="text-neutral-400">
            {completedCount}/{totalCount}
          </Text>
          <SpinnerIcon />
        </button>
      ) : (
        <Button
          onClick={handleOpenDrawer}
          kind="tertiary"
          size="tiny"
        >
          <MoreIcon />
        </Button>
      )}
    </Flex>
  );
}; 
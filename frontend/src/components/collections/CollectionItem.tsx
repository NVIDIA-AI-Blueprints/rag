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
  const hasPendingTasks = pendingTasks.some(
    (t) => t.collection_name === collection.collection_name && t.state === "PENDING"
  );

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
      {hasPendingTasks ? <SpinnerIcon /> : (
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
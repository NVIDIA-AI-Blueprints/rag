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
import { useCollectionsStore } from "../../store/useCollectionsStore";
import { 
  Stack, 
  Flex, 
  Text, 
  Tag 
} from "@kui/react";

/**
 * Icon component for collection representation.
 */
const CollectionIcon = () => (
  <svg 
    style={{ width: '12px', height: '12px', color: 'var(--text-color-subtle)' }} 
    fill="none" 
    stroke="currentColor" 
    strokeWidth="2" 
    viewBox="0 0 24 24"
  >
    <path strokeLinecap="round" strokeLinejoin="round" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
  </svg>
);

const RemoveIcon = () => (
  <svg 
    style={{ width: '12px', height: '12px' }} 
    fill="none" 
    stroke="currentColor" 
    strokeWidth="2" 
    viewBox="0 0 24 24"
  >
    <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
  </svg>
);

interface CollectionChipProps {
  name: string;
  onRemove: () => void;
}

const CollectionChip = ({ name, onRemove }: CollectionChipProps) => (
  <Tag
    color="gray"
    kind="outline"
    density="compact"
    onClick={onRemove}
    title="Remove collection"
  >
    <Flex align="center" gap="density-xs">
      <Text kind="body/bold/xs">{name}</Text>
      <RemoveIcon />
    </Flex>
  </Tag>
);

/**
 * Component that displays selected collections as removable chips.
 * 
 * Shows a horizontal list of selected collection names as chips with
 * remove buttons. Allows users to deselect collections by clicking
 * the remove icon on each chip.
 * 
 * @returns Collection chips component with remove functionality
 */
export const CollectionChips = () => {
  const { selectedCollections, toggleCollection } = useCollectionsStore();

  const handleRemove = useCallback((name: string) => {
    toggleCollection(name);
  }, [toggleCollection]);

  if (selectedCollections.length === 0) {
    return null;
  }

  return (
    <Stack>
      <Flex align="center" gap="density-sm" style={{ flexWrap: 'wrap' }}>
        <Flex align="center" gap="density-xs">
          <CollectionIcon />
          <Text kind="label/regular/md" style={{ color: 'var(--text-color-subtle)' }}>
            Collections:
          </Text>
        </Flex>
        {selectedCollections.map((name) => (
          <CollectionChip
            key={name}
            name={name}
            onRemove={() => handleRemove(name)}
          />
        ))}
      </Flex>
    </Stack>
  );
}; 
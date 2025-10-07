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

// src/components/MessageInput.tsx
import { useChatStore } from "../../store/useChatStore";
import { useCollectionsStore } from "../../store/useCollectionsStore";
import { CollectionChips } from "../collections/CollectionChips";
import { MessageInputContainer } from "./MessageInputContainer";
import SimpleFilterBar from "../filtering/SimpleFilterBar";
import { Flex, Banner, Block } from "@kui/react";

// Export all message input components for external use
export { CollectionChips } from "../collections/CollectionChips";
export { MessageTextarea } from "./MessageTextarea";
export { MessageActions } from "./MessageActions";
export { MessageInputContainer } from "./MessageInputContainer";

export default function MessageInput() {
  const { filters, setFilters } = useChatStore();
  const { selectedCollections } = useCollectionsStore();

  // Debug: Check if MessageInput renders multiple times
  console.log('💬 MessageInput RENDER:', {
    selectedCollections: selectedCollections.length,
    filters: filters.length,
    timestamp: Date.now()
  });

  return (
    <Flex direction="col" padding="density-sm">
      <CollectionChips />
      
      <>
        {selectedCollections.length === 1 && (
          <SimpleFilterBar filters={filters} setFilters={setFilters} />
        )}
        {selectedCollections.length > 1 && (
          <Block paddingY="density-sm">
            <Banner status="warning" kind="inline">
              Filters not available with more than one collection selected
            </Banner>
          </Block>
        )}
        <MessageInputContainer />
      </>
    </Flex>
  );
}

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

import { useState } from "react";
import { useCollections } from "../../api/useCollectionsApi";
import { CollectionsGrid } from "./CollectionsGrid";
import { NewCollectionButton } from "./NewCollectionButton";
import CollectionDrawer from "./CollectionDrawer";
import { Block, TextInput } from "@kui/react";
export { CollectionItem } from "./CollectionItem";
export { CollectionsGrid } from "./CollectionsGrid";
export { NewCollectionButton } from "./NewCollectionButton";

export default function CollectionList() {
  const { isLoading } = useCollections();
  const [searchQuery, setSearchQuery] = useState("");

  return (
    <>
      {/* CSS Grid layout - flexible and maintainable */}
      <Block 
        style={{ 
          height: '100%', 
          width: '100%',
          display: 'grid',
          gridTemplateRows: 'auto 1fr auto',
          gridTemplateColumns: '1fr'
        }}
      >
        {/* Search input - auto-sized row */}
        <Block 
          padding="density-sm" 
          style={{
            backgroundColor: 'var(--background-color-surface-navigation)',
            borderRight: '1px solid var(--border-color-base)'
          }}
        >
          <TextInput
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search collections"
          />
        </Block>

        {/* Scrollable collections - takes remaining space (1fr) */}
        <Block style={{
          overflowY: 'auto',
          minHeight: 0 // Prevents grid item from growing beyond container
        }}>
          <CollectionsGrid searchQuery={searchQuery} />
        </Block>

        {/* Add collection button - auto-sized row */}
        <Block 
          padding="density-sm"
          style={{ 
            backgroundColor: 'var(--background-color-surface-navigation)',
            borderRight: '1px solid var(--border-color-base)'
          }}
        >
          <NewCollectionButton disabled={isLoading} />
        </Block>
      </Block>

      <CollectionDrawer />
    </>
  );
}

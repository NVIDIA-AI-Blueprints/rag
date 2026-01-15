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

import { Text, Flex, Stack, Badge, Tag, Divider, Panel } from "@kui/react";
import { User, Building2, Calendar, FileText, Table, BarChart3, Image } from "lucide-react";
import type { Collection } from "../../types/collections";

interface CollectionCatalogInfoProps {
  collection: Collection;
  /** Optional override for file count - use actual document count from API for accuracy */
  documentCount?: number;
}

export function CollectionCatalogInfo({ collection, documentCount }: CollectionCatalogInfoProps) {
  const info = collection.collection_info;
  
  if (!info) {
    return null;
  }
  
  // Use actual document count if provided, otherwise fall back to collection_info
  const fileCount = documentCount ?? info.number_of_files;

  const formatDate = (dateString?: string) => {
    if (!dateString) return 'N/A';
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const getStatusColor = (status?: string): "green" | "yellow" | "red" | "gray" => {
    switch (status) {
      case 'Active': return 'green';
      case 'Archived': return 'yellow';
      case 'Deprecated': return 'red';
      default: return 'gray';
    }
  };

  return (
    <Panel elevation="high" density="compact">
      <Stack gap="density-md" style={{ padding: 'var(--spacing-density-md)' }}>
        {/* Description */}
        {info.description && (
          <div>
            <Text kind="body/regular/sm">
              {info.description}
            </Text>
          </div>
        )}

        {/* Tags */}
        {info.tags && info.tags.length > 0 && (
          <Flex gap="density-xs" style={{ flexWrap: 'wrap' }}>
            {info.tags.map((tag) => (
              <Tag key={tag} color="gray" kind="outline" density="compact" readOnly>
                {tag}
              </Tag>
            ))}
          </Flex>
        )}

        <Divider />

        {/* Metadata Grid */}
        <Flex gap="density-lg" style={{ flexWrap: 'wrap' }}>
          {/* Owner */}
          {info.owner && (
            <Flex align="center" gap="density-xs">
              <User size={14} style={{ color: 'var(--text-color-subtle)' }} />
              <Text kind="body/regular/sm" style={{ color: 'var(--text-color-subtle)' }}>
                {info.owner}
              </Text>
            </Flex>
          )}

          {/* Business Domain */}
          {info.business_domain && (
            <Flex align="center" gap="density-xs">
              <Building2 size={14} style={{ color: 'var(--text-color-subtle)' }} />
              <Text kind="body/regular/sm" style={{ color: 'var(--text-color-subtle)' }}>
                {info.business_domain}
              </Text>
            </Flex>
          )}

          {/* Status */}
          {info.status && (
            <Badge color={getStatusColor(info.status)} kind="solid">
              {info.status}
            </Badge>
          )}
        </Flex>

        {/* Dates */}
        {(info.date_created || info.last_updated) && (
          <Flex gap="density-lg" style={{ flexWrap: 'wrap' }}>
            {info.date_created && (
              <Flex align="center" gap="density-xs">
                <Calendar size={14} style={{ color: 'var(--text-color-subtle)' }} />
                <Text kind="body/regular/sm" style={{ color: 'var(--text-color-subtle)' }}>
                  Created: {formatDate(info.date_created)}
                </Text>
              </Flex>
            )}
            {info.last_updated && (
              <Text kind="body/regular/sm" style={{ color: 'var(--text-color-subtle)' }}>
                Updated: {formatDate(info.last_updated)}
              </Text>
            )}
          </Flex>
        )}

        {/* Content Metrics */}
        <Flex gap="density-md" style={{ flexWrap: 'wrap' }}>
          {/* File Count - use actual document count if available */}
          {fileCount !== undefined && (
            <Flex align="center" gap="density-xs">
              <FileText size={14} style={{ color: 'var(--text-color-subtle)' }} />
              <Text kind="body/regular/sm">
                {fileCount} files
              </Text>
            </Flex>
          )}

          {/* Content Type Counts */}
          {info.doc_type_counts?.table !== undefined && info.doc_type_counts.table > 0 && (
            <Flex align="center" gap="density-xs">
              <Table size={14} style={{ color: 'var(--color-brand)' }} />
              <Text kind="body/regular/sm">{info.doc_type_counts.table} Tables</Text>
            </Flex>
          )}

          {info.doc_type_counts?.chart !== undefined && info.doc_type_counts.chart > 0 && (
            <Flex align="center" gap="density-xs">
              <BarChart3 size={14} style={{ color: 'var(--color-brand)' }} />
              <Text kind="body/regular/sm">{info.doc_type_counts.chart} Charts</Text>
            </Flex>
          )}

          {info.doc_type_counts?.image !== undefined && info.doc_type_counts.image > 0 && (
            <Flex align="center" gap="density-xs">
              <Image size={14} style={{ color: 'var(--color-brand)' }} />
              <Text kind="body/regular/sm">{info.doc_type_counts.image} Images</Text>
            </Flex>
          )}
        </Flex>
      </Stack>
    </Panel>
  );
}


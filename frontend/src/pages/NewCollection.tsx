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

import { useEffect, useCallback, useState } from "react";
import NvidiaUpload from "../components/files/NvidiaUpload";
import MetadataSchemaEditor from "../components/schema/MetadataSchemaEditor";
import NewCollectionButtons from "../components/collections/NewCollectionButtons";
import { CollectionConfigurationPanel } from "../components/collections/CollectionConfigurationPanel";
import { useNewCollectionStore } from "../store/useNewCollectionStore";
import { Block, FormField, Grid, GridItem, PageHeader, Panel, Stack, TextInput, Select, Text, Tag, Flex } from "@kui/react";
import { X, ChevronDown, BookOpen } from "lucide-react";

/**
 * New Collection page component for creating collections.
 * 
 * Provides a multi-step interface for collection creation including
 * file upload, metadata schema definition, and collection naming.
 * 
 * @returns The new collection page component
 */
// Business domain options
const BUSINESS_DOMAINS = [
  'Engineering',
  'Finance', 
  'Legal',
  'Marketing',
  'Operations',
  'Product',
  'Sales',
  'Support',
  'Other'
];

// Status options
const STATUS_OPTIONS = ['Active', 'Archived', 'Deprecated'];

// Catalog Metadata Section Component
interface CatalogMetadataSectionProps {
  catalogMetadata: {
    description: string;
    tags: string[];
    owner: string;
    business_domain: string;
    status: 'Active' | 'Archived' | 'Deprecated';
  };
  setCatalogMetadata: (updates: Partial<CatalogMetadataSectionProps['catalogMetadata']>) => void;
  onAddTag: (tag: string) => void;
  onRemoveTag: (tag: string) => void;
}

function CatalogMetadataSection({ 
  catalogMetadata, 
  setCatalogMetadata, 
  onAddTag, 
  onRemoveTag 
}: CatalogMetadataSectionProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [tagInput, setTagInput] = useState('');

  return (
    <Panel
      slotHeading={
        <Flex 
          align="center" 
          justify="between" 
          style={{ width: '100%', cursor: 'pointer' }}
          onClick={() => setIsExpanded(!isExpanded)}
        >
          <span>Data Catalog</span>
          <ChevronDown 
            size={16} 
            style={{ 
              transform: isExpanded ? 'rotate(180deg)' : 'rotate(0deg)',
              transition: 'transform 0.2s ease'
            }} 
          />
        </Flex>
      }
      slotIcon={<BookOpen size={20} />}
    >
      <Text kind="body/bold/md">
        Optional metadata for organizing, categorizing, and governing your collections.
      </Text>

      {isExpanded && (
        <Stack gap="density-md" style={{ marginTop: 'var(--spacing-density-lg)' }}>
          <FormField
            slotLabel="Description"
            slotHelp="Human-readable description of the collection."
          >
            <TextInput
              value={catalogMetadata.description}
              onValueChange={(value) => setCatalogMetadata({ description: value })}
              placeholder="e.g., Q4 2024 Financial Reports"
            />
          </FormField>

          <FormField
            slotLabel="Tags"
            slotHelp="Tags for categorization and discovery. Press Enter to add."
          >
            <Stack gap="density-sm">
              <TextInput
                placeholder="Add a tag and press Enter"
                value={tagInput}
                onValueChange={setTagInput}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    e.preventDefault();
                    if (tagInput.trim()) {
                      onAddTag(tagInput.trim());
                      setTagInput('');
                    }
                  }
                }}
              />
              {catalogMetadata.tags.length > 0 && (
                <Flex gap="density-xs" style={{ flexWrap: 'wrap' }}>
                  {catalogMetadata.tags.map((tag) => (
                    <Tag
                      key={tag}
                      color="gray"
                      kind="outline"
                      density="compact"
                      onClick={() => onRemoveTag(tag)}
                      style={{ cursor: 'pointer' }}
                    >
                      {tag} <X size={12} />
                    </Tag>
                  ))}
                </Flex>
              )}
            </Stack>
          </FormField>

          <FormField
            slotLabel="Owner"
            slotHelp="Team or person responsible for this collection."
          >
            <TextInput
              value={catalogMetadata.owner}
              onValueChange={(value) => setCatalogMetadata({ owner: value })}
              placeholder="e.g., Finance Team"
            />
          </FormField>

          <FormField
            slotLabel="Business Domain"
            slotHelp="Business domain or department."
          >
            <Select
              items={BUSINESS_DOMAINS}
              value={catalogMetadata.business_domain}
              onValueChange={(value) => setCatalogMetadata({ business_domain: value })}
              placeholder="Select a domain"
            />
          </FormField>

          <FormField
            slotLabel="Status"
            slotHelp="Collection lifecycle status."
          >
            <Select
              items={STATUS_OPTIONS}
              value={catalogMetadata.status}
              onValueChange={(value) => setCatalogMetadata({ status: value as 'Active' | 'Archived' | 'Deprecated' })}
            />
          </FormField>
        </Stack>
      )}
    </Panel>
  );
}

export default function NewCollection() {
  const { 
    collectionName, 
    setCollectionName, 
    setCollectionNameTouched, 
    catalogMetadata,
    setCatalogMetadata,
    collectionConfig,
    setCollectionConfig,
    reset 
  } = useNewCollectionStore();

  useEffect(() => {
    // cleanup when leaving the page
    return () => {
      reset();
    };
  }, [reset]);

  const handleValidationChange = useCallback((hasInvalidFiles: boolean) => {
    const { setHasInvalidFiles } = useNewCollectionStore.getState();
    setHasInvalidFiles(hasInvalidFiles);
  }, []);

  const handleFilesChange = useCallback((files: File[]) => {
    const { setFiles } = useNewCollectionStore.getState();
    setFiles(files);
  }, []);

  const handleAddTag = useCallback((tag: string) => {
    if (tag.trim() && !catalogMetadata.tags.includes(tag.trim())) {
      setCatalogMetadata({ tags: [...catalogMetadata.tags, tag.trim()] });
    }
  }, [catalogMetadata.tags, setCatalogMetadata]);

  const handleRemoveTag = useCallback((tagToRemove: string) => {
    setCatalogMetadata({ 
      tags: catalogMetadata.tags.filter(t => t !== tagToRemove) 
    });
  }, [catalogMetadata.tags, setCatalogMetadata]);

  return (
    <Grid cols={12} gap="density-lg" padding="density-lg">
      <GridItem cols={12}>
        <Block padding="density-lg">
          <PageHeader
            slotHeading="Create New Collection"
            slotSubheading="Upload source files and define metadata schema for this collection."
          />
        </Block>
      </GridItem>
      <GridItem cols={6}>
        <Panel>
          <Stack gap="density-lg">
            <FormField
              slotLabel="Collection Name"
              slotHelp="We will automatically try to validate the collection name."
              required
            >
              <TextInput
                value={collectionName}
                onChange={(e) => setCollectionName(e.target.value.replace(/\s+/g, "_"))}
                onBlur={() => setCollectionNameTouched(true)}
              />
            </FormField>

            {/* Catalog Metadata Section */}
            <CatalogMetadataSection 
              catalogMetadata={catalogMetadata}
              setCatalogMetadata={setCatalogMetadata}
              onAddTag={handleAddTag}
              onRemoveTag={handleRemoveTag}
            />

            {/* Collection Configuration Section */}
            <CollectionConfigurationPanel
              generateSummary={collectionConfig.generateSummary}
              onGenerateSummaryChange={(value) => setCollectionConfig({ generateSummary: value })}
            />

            <MetadataSchemaEditor />
          </Stack>
        </Panel>
      </GridItem>
      <GridItem cols={6}>
        <Panel>
          <Stack 
            gap="density-xl"
            style={{ 
              borderTop: '1px solid var(--border-color-subtle)',
            }}
          >
            <NvidiaUpload 
              onFilesChange={handleFilesChange}
              onValidationChange={handleValidationChange}
              acceptedTypes={['.bmp', '.docx', '.html', '.jpeg', '.json', '.md', '.pdf', '.png', '.pptx', '.sh', '.tiff', '.txt', '.mp3', '.wav', '.mp4', '.mov', '.avi']}
              maxFileSize={400}
            />
          </Stack>
        </Panel>
      </GridItem>
      <GridItem cols={12}>
        <NewCollectionButtons />
      </GridItem>
    </Grid>
  );
}
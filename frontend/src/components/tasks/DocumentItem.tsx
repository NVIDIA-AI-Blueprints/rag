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
import { useFileIcons } from "../../hooks/useFileIcons";
import { useDeleteDocument, useUpdateDocumentMetadata } from "../../api/useCollectionDocuments";
import { useDocumentSummary } from "../../api/useDocumentSummary";
import { useCollectionDrawerStore } from "../../store/useCollectionDrawerStore";
import { useQueryClient } from "@tanstack/react-query";
import type { Collection } from "../../types/collections";
import { ConfirmationModal } from "../modals/ConfirmationModal";
import { 
  Flex, 
  Stack, 
  Text, 
  Button,
  Spinner,
  Badge,
  TextInput,
  Modal,
  Tag,
  FormField
} from "@kui/react";
import { Trash2, ChevronDown, Pencil, X } from "lucide-react";
import type { DocumentInfo } from "../../types/api";

interface DocumentItemProps {
  name: string;
  metadata: Record<string, unknown>;
  collectionName: string;
  documentInfo?: DocumentInfo;
}

/**
 * Format file size in bytes to human readable string.
 */
const formatFileSize = (bytes: number): string => {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
};

// Helper function to format metadata values for display
const formatMetadataValue = (value: unknown): string => {
  if (value === null || value === undefined) {
    return "—";
  }
  
  if (typeof value === "boolean") {
    return value ? "true" : "false";
  }
  
  if (typeof value === "string") {
    // Handle string representations of booleans
    const lowerValue = value.toLowerCase().trim();
    if (lowerValue === "true" || lowerValue === "1" || lowerValue === "yes" || lowerValue === "on") {
      return "true";
    }
    if (lowerValue === "false" || lowerValue === "0" || lowerValue === "no" || lowerValue === "off") {
      return "false";
    }
    // Return the string as-is if it's not empty
    return value.trim() || "—";
  }
  
  return String(value);
};

/**
 * Display document summary with loading/error states.
 */
const DocumentSummary = ({ collectionName, fileName }: { collectionName: string; fileName: string }) => {
  const { data, isLoading } = useDocumentSummary(collectionName, fileName);
  const [expanded, setExpanded] = useState(false);

  // Don't show anything if no summary or not found
  if (!data || data.status === "NOT_FOUND" || data.status === "FAILED") {
    return null;
  }

  // Show loading state for pending/in-progress
  if (isLoading || data.status === "PENDING" || data.status === "IN_PROGRESS") {
    return (
      <Text kind="body/regular/sm">
        Generating summary...
      </Text>
    );
  }

  // Show summary if available
  if (data.status === "SUCCESS" && data.summary) {
    return (
      <div 
        onClick={() => setExpanded(!expanded)} 
        style={{ cursor: 'pointer' }}
      >
        <Flex gap="density-sm" align="start">
          <div
            style={expanded ? { flex: 1 } : {
              flex: 1,
              display: '-webkit-box',
              WebkitLineClamp: 2,
              WebkitBoxOrient: 'vertical',
              overflow: 'hidden',
            }}
          >
            <Text kind="body/regular/sm">
              {data.summary}
            </Text>
          </div>
          <ChevronDown 
            size={16} 
            style={{ 
              flexShrink: 0,
              marginTop: '12px',
              transform: expanded ? 'rotate(180deg)' : 'rotate(0deg)',
              transition: 'transform 0.2s ease',
            }} 
          />
        </Flex>
      </div>
    );
  }

  return null;
};

export const DocumentItem = ({ name, metadata, collectionName, documentInfo }: DocumentItemProps) => {
  const { getFileIconByExtension } = useFileIcons();
  const queryClient = useQueryClient();
  const { setDeleteError, updateActiveCollection } = useCollectionDrawerStore();
  const deleteDoc = useDeleteDocument();
  const updateMetadata = useUpdateDocumentMetadata();
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [showEditModal, setShowEditModal] = useState(false);
  const [editDescription, setEditDescription] = useState(documentInfo?.description || "");
  const [editTags, setEditTags] = useState<string[]>(documentInfo?.tags || []);
  const [newTag, setNewTag] = useState("");

  const handleDeleteClick = () => {
    if (!collectionName) return;
    setShowDeleteModal(true);
  };

  const handleEditClick = () => {
    setEditDescription(documentInfo?.description || "");
    setEditTags(documentInfo?.tags || []);
    setShowEditModal(true);
  };

  const handleAddTag = () => {
    const trimmedTag = newTag.trim();
    if (trimmedTag && !editTags.includes(trimmedTag)) {
      setEditTags([...editTags, trimmedTag]);
      setNewTag("");
    }
  };

  const handleRemoveTag = (tagToRemove: string) => {
    setEditTags(editTags.filter(tag => tag !== tagToRemove));
  };

  const handleTagKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      e.preventDefault();
      handleAddTag();
    }
  };

  const handleSaveMetadata = () => {
    updateMetadata.mutate(
      {
        collectionName,
        documentName: name,
        description: editDescription || undefined,
        tags: editTags.length > 0 ? editTags : undefined,
      },
      {
        onSuccess: () => {
          queryClient.invalidateQueries({ queryKey: ["collection-documents", collectionName] });
          setShowEditModal(false);
        },
      }
    );
  };

  const handleConfirmDelete = () => {
    setDeleteError(null);
    deleteDoc.mutate(
      { collectionName, documentName: name },
      {
        onSuccess: async () => {
          queryClient.invalidateQueries({ queryKey: ["collection-documents", collectionName] });
          // Refresh collections and update drawer with new collection_info
          await queryClient.refetchQueries({ queryKey: ["collections"] });
          const data = queryClient.getQueryData<Collection[]>(["collections"]);
          const fresh = data?.find((c: Collection) => c.collection_name === collectionName);
          if (fresh) updateActiveCollection(fresh);
        },
        onError: (err: Error) => {
          setDeleteError(err?.message || "Failed to delete document");
        },
      }
    );
  };
  
  return (
    <Stack data-testid="document-item" gap="density-md">
      {/* Header row: icon, name, delete button */}
      <Flex justify="between" align="center">
        <Flex align="center" gap="density-md">
          <div data-testid="document-icon">
            {getFileIconByExtension(name, { size: 'sm' })}
          </div>
          <Text kind="body/bold/md" data-testid="document-name">
            {name}
          </Text>
        </Flex>
        
        {/* Action buttons */}
        <Flex gap="density-xs">
          <Button
            kind="tertiary"
            size="tiny"
            onClick={handleEditClick}
            aria-label={`Edit ${name} metadata`}
            title="Edit metadata"
          >
            <Pencil size={16} />
          </Button>
          <Button
            kind="tertiary"
            size="tiny"
            onClick={handleDeleteClick}
            disabled={deleteDoc.isPending}
            aria-label={`Delete ${name}`}
            title="Delete"
          >
            {deleteDoc.isPending ? (
              <Spinner size="small" description="" />
            ) : (
              <Trash2 size={16} />
            )}
          </Button>
        </Flex>
      </Flex>
      
      {/* Document Info Badges */}
      {documentInfo && (
        <Flex gap="density-sm" wrap="wrap">
          {documentInfo.file_size && (
            <Badge kind="outline" color="gray">{formatFileSize(documentInfo.file_size)}</Badge>
          )}
          {documentInfo.doc_type_counts?.text && documentInfo.doc_type_counts.text > 0 && (
            <Badge kind="outline" color="gray">{documentInfo.doc_type_counts.text} text</Badge>
          )}
          {documentInfo.doc_type_counts?.table && documentInfo.doc_type_counts.table > 0 && (
            <Badge kind="outline" color="gray">{documentInfo.doc_type_counts.table} tables</Badge>
          )}
          {documentInfo.doc_type_counts?.chart && documentInfo.doc_type_counts.chart > 0 && (
            <Badge kind="outline" color="gray">{documentInfo.doc_type_counts.chart} charts</Badge>
          )}
        </Flex>
      )}

      {/* Document Description */}
      {documentInfo?.description && (
        <Text kind="body/regular/sm" style={{ color: 'var(--text-color-subtle)' }}>
          {documentInfo.description}
        </Text>
      )}

      {/* Document Tags */}
      {documentInfo?.tags && documentInfo.tags.length > 0 && (
        <Flex gap="density-xs" wrap="wrap">
          {documentInfo.tags.map((tag) => (
            <Tag key={tag} color="gray" kind="outline" density="compact" readOnly>
              {tag}
            </Tag>
          ))}
        </Flex>
      )}
      
      {/* Metadata */}
      {Object.keys(metadata).filter(key => key !== 'filename').length > 0 && (
        <Stack gap="1" data-testid="document-metadata">
          {Object.entries(metadata)
            .filter(([key]) => key !== 'filename')
            .map(([key, val]) => (
              <Flex key={key} gap="2" wrap="wrap">
                <Text kind="body/bold/sm">
                  {key}:
                </Text>
                <Text kind="body/regular/sm">
                  {formatMetadataValue(val)}
                </Text>
              </Flex>
            ))}
        </Stack>
      )}
      
      {/* Summary */}
      <DocumentSummary collectionName={collectionName} fileName={name} />

      <ConfirmationModal
        isOpen={showDeleteModal}
        onClose={() => setShowDeleteModal(false)}
        onConfirm={handleConfirmDelete}
        title="Delete Document"
        message={`Are you sure you want to delete "${name}"? This action cannot be undone.`}
        confirmText="Delete"
        confirmColor="danger"
      />

      {/* Edit Metadata Modal */}
      <Modal
        open={showEditModal}
        onOpenChange={() => setShowEditModal(false)}
        slotHeading="Edit Document Info"
        slotFooter={
          <Flex gap="density-sm" justify="end">
            <Button kind="secondary" onClick={() => setShowEditModal(false)}>
              Cancel
            </Button>
            <Button 
              kind="primary" 
              onClick={handleSaveMetadata}
              disabled={updateMetadata.isPending}
            >
              {updateMetadata.isPending ? <Spinner size="small" description="" /> : "Save"}
            </Button>
          </Flex>
        }
      >
        <Stack gap="density-md">
          <FormField slotLabel="Description">
            <TextInput
              value={editDescription}
              onValueChange={setEditDescription}
              placeholder="Enter a description for this document"
            />
          </FormField>
          
          <Stack gap="density-sm">
            <Text kind="body/bold/sm">Tags</Text>
            <Flex gap="density-xs" wrap="wrap">
              {editTags.map((tag) => (
                <Tag 
                  key={tag} 
                  color="gray" 
                  kind="outline" 
                  density="compact"
                  onClick={() => handleRemoveTag(tag)}
                >
                  <Flex align="center" gap="density-xs">
                    {tag}
                    <X size={12} />
                  </Flex>
                </Tag>
              ))}
            </Flex>
            <Flex gap="density-sm" align="end">
              <div style={{ flex: 1 }}>
                <TextInput
                  value={newTag}
                  onValueChange={setNewTag}
                  onKeyDown={handleTagKeyDown}
                  placeholder="Add a tag and press Enter"
                />
              </div>
              <Button kind="secondary" size="small" onClick={handleAddTag} disabled={!newTag.trim()}>
                Add
              </Button>
            </Flex>
          </Stack>
        </Stack>
      </Modal>
    </Stack>
  );
};

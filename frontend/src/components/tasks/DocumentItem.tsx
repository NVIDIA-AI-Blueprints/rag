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
import { useDeleteDocument } from "../../api/useCollectionDocuments";
import { useCollectionDrawerStore } from "../../store/useCollectionDrawerStore";
import { useQueryClient } from "@tanstack/react-query";
import { ConfirmationModal } from "../modals/ConfirmationModal";
import { 
  Flex, 
  Stack, 
  Text, 
  Button,
  Spinner 
} from "@kui/react";

const DeleteIcon = () => (
  <svg 
    style={{ width: '16px', height: '16px' }}
    fill="none" 
    stroke="currentColor" 
    strokeWidth="2" 
    viewBox="0 0 24 24"
  >
    <path strokeLinecap="round" strokeLinejoin="round" d="m14.74 9-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 0 1-2.244 2.077H8.084a2.25 2.25 0 0 1-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 0 0-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 0 1 3.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 0 0-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 0 0-7.5 0" />
  </svg>
);

interface DocumentItemProps {
  name: string;
  metadata: Record<string, unknown>;
  collectionName: string;
}

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

export const DocumentItem = ({ name, metadata, collectionName }: DocumentItemProps) => {
  const { getFileIconByExtension } = useFileIcons();
  const queryClient = useQueryClient();
  const { setDeleteError } = useCollectionDrawerStore();
  const deleteDoc = useDeleteDocument();
  const [showDeleteModal, setShowDeleteModal] = useState(false);

  const handleDeleteClick = () => {
    if (!collectionName) return;
    setShowDeleteModal(true);
  };

  const handleConfirmDelete = () => {
    setDeleteError(null);
    deleteDoc.mutate(
      { collectionName, documentName: name },
      {
        onSuccess: () => {
          queryClient.invalidateQueries({ queryKey: ["collection-documents", collectionName] });
        },
        onError: (err: Error) => {
          setDeleteError(err?.message || "Failed to delete document");
        },
      }
    );
  };
  
  return (
    <Stack data-testid="document-item">
      <Flex justify="between" align="start">
        <Stack gap="density-md">
          {/* Document name and icon */}
          <Flex align="center" gap="density-md">
            <div data-testid="document-icon">
              {getFileIconByExtension(name, { size: 'sm' })}
            </div>
            <Text  kind="body/bold/md" style={{ color: 'var(--text-color-inverse)' }} data-testid="document-name">
              {name}
            </Text>
          </Flex>
          
          {/* Metadata */}
          {Object.keys(metadata).filter(key => key !== 'filename').length > 0 && (
            <Stack gap="1" data-testid="document-metadata">
              {Object.entries(metadata)
                .filter(([key]) => key !== 'filename')
                .map(([key, val]) => (
                  <Flex key={key} gap="2" wrap="wrap">
                    <Text kind="body/bold/sm" style={{ color: 'var(--text-color-inverse)' }}>
                      {key}:
                    </Text>
                    <Text kind="body/regular/sm" style={{ color: 'var(--text-color-inverse)' }}>
                      {formatMetadataValue(val)}
                    </Text>
                  </Flex>
                ))}
            </Stack>
          )}
        </Stack>
        
        {/* Delete button */}
        <Button
          kind="tertiary"
          size="tiny"
          color="danger"
          onClick={handleDeleteClick}
          disabled={deleteDoc.isPending}
          aria-label={`Delete ${name}`}
          title="Delete"
        >
          {deleteDoc.isPending ? (
            <Spinner size="small" description="" />
          ) : (
            <DeleteIcon />
          )}
        </Button>
      </Flex>

      <ConfirmationModal
        isOpen={showDeleteModal}
        onClose={() => setShowDeleteModal(false)}
        onConfirm={handleConfirmDelete}
        title="Delete Document"
        message={`Are you sure you want to delete "${name}"? This action cannot be undone.`}
        confirmText="Delete"
        confirmColor="danger"
      />
    </Stack>
  );
}; 
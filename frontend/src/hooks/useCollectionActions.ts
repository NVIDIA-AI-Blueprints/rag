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

import { useNavigate } from "react-router-dom";
import { useQueryClient } from "@tanstack/react-query";
import { useDeleteCollection } from "../api/useCollectionsApi";
import { useDeleteAllDocuments } from "../api/useCollectionDocuments";
import { useCollectionsStore } from "../store/useCollectionsStore";
import { useNotificationStore } from "../store/useNotificationStore";
import { useUploadDocuments } from "./useUploadDocuments";
import { useNewCollectionStore } from "../store/useNewCollectionStore";
import { useCollectionDrawerStore } from "../store/useCollectionDrawerStore";
import { openNotificationPanel } from "../components/notifications/NotificationBell";

/**
 * Custom hook for managing collection-related actions and operations.
 * 
 * Provides comprehensive collection management functionality including deletion,
 * document upload, bulk operations, and UI state management. Handles navigation,
 * notifications, and error states for collection operations.
 * 
 * @returns Object with collection action functions and state management
 * 
 * @example
 * ```tsx
 * const { handleDeleteCollection, handleUploadFiles, handleBulkDelete } = useCollectionActions();
 * await handleDeleteCollection("my-collection");
 * ```
 */
export function useCollectionActions() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { selectedCollections, clearCollections } = useCollectionsStore();
  const { addTaskNotification, updateTaskNotification } = useNotificationStore();
  const { selectedFiles, fileMetadata, reset } = useNewCollectionStore();
  const { activeCollection, closeDrawer, toggleUploader } = useCollectionDrawerStore();
  const deleteCollection = useDeleteCollection();
  const deleteAllDocuments = useDeleteAllDocuments();
  const uploadDocuments = useUploadDocuments();

  const handleDeleteCollection = async (collectionName: string) => {
    if (!window.confirm(`Are you sure you want to delete "${collectionName}"?`)) {
      return;
    }

    deleteCollectionWithoutConfirm(collectionName);
  };

  // Delete collection without confirmation modal (for use with custom confirmation UI)
  const deleteCollectionWithoutConfirm = (collectionName: string) => {
    deleteCollection.mutate(collectionName, {
      onSuccess: () => {
        queryClient.invalidateQueries({ queryKey: ["collections"] });
        
        // Close the drawer if the deleted collection was the active one
        if (activeCollection?.collection_name === collectionName) {
          closeDrawer();
        }
        
        // If this was a selected collection, clear selections
        if (selectedCollections.includes(collectionName)) {
          clearCollections();
          navigate("/");
        }
      },
    });
  };

  const handleDeleteAllDocuments = async (collectionName: string) => {
    if (!window.confirm(`Are you sure you want to delete all documents in "${collectionName}"?`)) {
      return;
    }

    const taskId = `delete-all-${Date.now()}`;
    
    // Add a pending task for the deletion
    addTaskNotification({
      id: taskId,
      collection_name: collectionName,
      documents: [],
      state: "PENDING",
      created_at: new Date().toISOString(),
    });

    deleteAllDocuments.mutate(collectionName, {
      onSuccess: () => {
        queryClient.invalidateQueries({ queryKey: ["collections"] });
        queryClient.invalidateQueries({ queryKey: ["documents", collectionName] });
        
        // Update task as completed
        setTimeout(() => {
          updateTaskNotification(taskId, {
            state: "FINISHED",
          });
          openNotificationPanel();
        }, 1000);
      },
      onError: () => {
        // Update task as failed
        updateTaskNotification(taskId, {
          state: "FAILED",
        });
      },
    });
  };

  const handleUploadDocuments = () => {
    if (!activeCollection?.collection_name || !selectedFiles.length) return;

    console.log("🚀 Starting add source upload:", { 
      collection: activeCollection.collection_name, 
      fileCount: selectedFiles.length 
    });

    // Helper function to process metadata values based on field type
    const processMetadataValue = (key: string, value: unknown) => {
      // Find the field definition in the collection's metadata schema
      const fieldDef = activeCollection?.metadata_schema?.find(f => f.name === key);
      
      if (fieldDef?.type === "array" && typeof value === "string") {
        try {
          // Parse JSON string back to array for array fields
          return JSON.parse(value);
        } catch {
          console.warn(`Failed to parse array value for field ${key}:`, value);
          return [];
        }
      }
      
      if (fieldDef?.type === "boolean") {
        // Handle boolean fields - convert string representations to actual booleans
        if (typeof value === "boolean") {
          return value;
        }
        
        if (typeof value === "string") {
          const lowerValue = value.toLowerCase().trim();
          if (lowerValue === "" || lowerValue === "false" || lowerValue === "0" || lowerValue === "no" || lowerValue === "off") {
            return false;
          }
          if (lowerValue === "true" || lowerValue === "1" || lowerValue === "yes" || lowerValue === "on") {
            return true;
          }
        }
        
        // Default to false for boolean fields if value is unclear
        return false;
      }
      
      // Handle numeric fields - convert empty strings to null or proper numbers
      if (fieldDef?.type === "integer") {
        if (typeof value === "string") {
          const trimmed = value.trim();
          if (trimmed === "") return null; // Empty string becomes null
          const num = parseInt(trimmed);
          if (isNaN(num)) return null;
          return num;
        }
        return value;
      }
      
      if (fieldDef?.type === "float" || fieldDef?.type === "number") {
        if (typeof value === "string") {
          const trimmed = value.trim();
          if (trimmed === "") return null; // Empty string becomes null
          const num = parseFloat(trimmed);
          if (isNaN(num)) return null;
          return num;
        }
        return value;
      }
      
      return value;
    };

    // Clean metadata by removing filename field and empty values
    const cleanedMetadata = selectedFiles.map((file) => {
      const rawMetadata = fileMetadata[file.name] || {};
      // Remove filename field if it exists and clean empty values
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
      const { filename: _filename, ...cleanedFields } = rawMetadata;
      const filtered = Object.fromEntries(
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        Object.entries(cleanedFields).filter(([_key, value]) => {
          // Handle undefined/null values - exclude them
          if (value === undefined || value === null) return false;
          
          // For actual boolean values (true/false), always keep them
          if (typeof value === "boolean") return true;
          
          // For string values, check if they're valid boolean representations
          if (typeof value === "string") {
            const trimmedValue = value.trim();
            if (trimmedValue === "") return false; // Empty strings are invalid
            
            const isValidBooleanString = ["true", "false", "1", "0", "yes", "no", "on", "off"]
              .includes(trimmedValue.toLowerCase());
            
            // Keep valid boolean strings, or non-empty non-boolean strings
            return isValidBooleanString || trimmedValue !== "";
          }
          
          // For numbers, keep all valid numbers (including 0)
          if (typeof value === "number" && !isNaN(value)) return true;
          
          // For other types (arrays, etc.), keep if truthy
          return !!value;
        }).map(([key, value]) => [key, processMetadataValue(key, value)])
      );
      
      return {
        filename: file.name,
        metadata: filtered,
      };
    });

    const metadata = {
      collection_name: activeCollection.collection_name,
      blocking: false,
      custom_metadata: cleanedMetadata,
      split_options: { chunk_size: 512, chunk_overlap: 150 },
      generate_summary: false,
    };

    uploadDocuments.mutate(
      { files: selectedFiles, metadata },
      {
        onSuccess: (data) => {
          console.log("✅ Add source upload successful:", data);
          
          // Reset upload state and close drawer immediately
          console.log("🧹 Cleaning up add source state and closing drawer");
          reset();
          toggleUploader(false);
          closeDrawer();
          
          // Open notification panel after successful upload
          setTimeout(() => {
            console.log("🔔 Opening notification panel for add source");
            openNotificationPanel();
          }, 100);
        },
        onError: (error) => {
          console.error("❌ Add source upload failed:", error);
        }
      }
    );
  };

  return {
    handleDeleteCollection,
    deleteCollectionWithoutConfirm,
    handleDeleteAllDocuments,
    handleUploadDocuments,
    isUploading: uploadDocuments.isPending,
    isDeleting: deleteCollection.isPending,
  };
} 
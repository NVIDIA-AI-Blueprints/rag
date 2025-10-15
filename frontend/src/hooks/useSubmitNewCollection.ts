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
import { useNewCollectionStore } from "../store/useNewCollectionStore";
import { useCreateCollection } from "../api/useCollectionsApi";
import { useNotificationStore } from "../store/useNotificationStore";
import { useSettingsStore } from "../store/useSettingsStore";
import { openNotificationPanel } from "../components/notifications/NotificationBell";
import { useQueryClient } from "@tanstack/react-query";
import type { APIMetadataField } from "../types/collections";
import type { CreateCollectionPayload } from "../types/api";

export function useSubmitNewCollection() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { vdbEndpoint } = useSettingsStore();
  const {
    collectionName,
    metadataSchema,
    fileMetadata,
    selectedFiles,
    setIsLoading,
    setUploadComplete,
    setError,
    reset,
  } = useNewCollectionStore();

  const { addTaskNotification } = useNotificationStore();
  const createCollection = useCreateCollection();

  const submit = async () => {
    console.log("🚀 Starting collection submission:", { collectionName, fileCount: selectedFiles.length });
    
    const filteredSchema = metadataSchema.map((field) => {
      const schemaField: APIMetadataField = { 
        name: field.name, 
        type: field.type,
        description: field.description || `${field.name} field`
      };
      
      // Include required field
      if (field.required !== undefined) {
        schemaField.required = field.required;
      }
      
      // Include array_type for array fields (required by backend)
      if (field.type === "array" && field.array_type) {
        schemaField.array_type = field.array_type;
      }
      
      // Include max_length if specified
      if (field.max_length !== undefined) {
        schemaField.max_length = field.max_length;
      }
      
      return schemaField;
    });
    
    const collectionPayload: CreateCollectionPayload = {
      collection_name: collectionName,
      metadata_schema: filteredSchema,
      embedding_dimension: 2048,
    };

    // Only include vdb_endpoint if explicitly set by user
    if (vdbEndpoint) {
      collectionPayload.vdb_endpoint = vdbEndpoint;
    }

    setIsLoading(true);
    setError(null);
    
    try {
      console.log("📝 Creating collection...");
      await new Promise((resolve, reject) => {
        createCollection.mutate(collectionPayload, {
          onSuccess: (data) => {
            console.log("✅ Collection created successfully:", data);
            resolve(data);
          },
          onError: (error) => {
            console.error("❌ Collection creation failed:", error);
            reject(error);
          },
        });
      });

      console.log("🔄 Invalidating collections cache...");
      await queryClient.invalidateQueries({ queryKey: ["collections"] });
      await queryClient.refetchQueries({ queryKey: ["collections"] });
      console.log("✅ Collections cache refreshed");

      if (selectedFiles.length > 0) {
        const formData = new FormData();
        selectedFiles.forEach((file) => {
          formData.append("documents", file);
        });

        // Helper function to process metadata values based on field type
        const processMetadataValue = (key: string, value: unknown) => {
          // Find the field definition in the metadata schema
          const fieldDef = metadataSchema.find(f => f.name === key);
          
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

        const metadata = {
          collection_name: collectionName,
          blocking: false,
          custom_metadata: selectedFiles.map((file) => {
            const rawFileMetadata = fileMetadata[file.name] || {};
            // Process metadata values to convert array JSON strings back to arrays and handle types
            const processedMetadata = Object.fromEntries(
              Object.entries(rawFileMetadata)
                .map(([key, value]) => [key, processMetadataValue(key, value)])
                // eslint-disable-next-line @typescript-eslint/no-unused-vars
                .filter(([_key, value]) => value !== null && value !== undefined) // Exclude null/undefined values
            );
            
            return {
              filename: file.name,
              metadata: processedMetadata,
            };
          }),
        };

        formData.append("data", JSON.stringify(metadata));

        const res = await fetch(`/api/documents?blocking=false`, {
          method: "POST",
          body: formData,
        });

        if (!res.ok) throw new Error("Failed to upload documents");

        const data = await res.json();
        console.log("📤 Upload response:", data);

        if (data?.task_id) {
          const taskData = {
            id: data.task_id,
            collection_name: collectionName,
            documents: selectedFiles.map((f) => f.name),
            state: "PENDING" as const,
            created_at: new Date().toISOString(),
          };
          
          console.log("📋 Adding pending task:", taskData);
          addTaskNotification(taskData);
          
          setTimeout(() => {
            console.log("🔔 Opening notification panel");
            openNotificationPanel();
          }, 100);
        }
      }

      console.log("🎉 Collection submission completed successfully");
      setUploadComplete(true);
      
      console.log("🔄 Resetting collection store state");
      reset(); 
      
      navigate("/");
    } catch (err: unknown) {
      console.error("💥 Collection submission failed:", err);
      setError(err instanceof Error ? err.message : "Failed to create collection");
    } finally {
      setIsLoading(false);
    }
  };

  return { submit };
}

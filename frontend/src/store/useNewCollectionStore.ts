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

import { create } from "zustand";
import type { UIMetadataField } from "../types/collections";

/**
 * Get the default value for a metadata field based on its type.
 */
const getDefaultValueForField = (field: UIMetadataField): unknown => {
  switch (field.type) {
    case "string":
    case "datetime":
      return "";
    case "integer":
    case "float":
    case "number":
      return null;
    case "boolean":
      return false;
    case "array":
      return [];
    default:
      return "";
  }
};

/**
 * State interface for the new collection creation flow.
 */
interface NewCollectionState {
  collectionName: string;
  collectionNameTouched: boolean;
  selectedFiles: File[];
  fileMetadata: Record<string, Record<string, unknown>>;
  metadataSchema: UIMetadataField[];
  isLoading: boolean;
  uploadComplete: boolean;
  error: string | null;
  hasInvalidFiles: boolean; // New state to track file validation
  setCollectionName: (name: string) => void;
  setCollectionNameTouched: (touched: boolean) => void;
  setMetadataSchema: (schema: UIMetadataField[]) => void;
  setIsLoading: (v: boolean) => void;
  setUploadComplete: (v: boolean) => void;
  setError: (msg: string | null) => void;
  setHasInvalidFiles: (hasInvalidFiles: boolean) => void; // New setter
  addFiles: (files: File[]) => void;
  setFiles: (files: File[]) => void; // Replace files instead of appending
  removeFile: (index: number) => void;
  updateMetadataField: (filename: string, field: string, value: unknown) => void;
  reset: () => void;
}

/**
 * Zustand store for managing the new collection creation process.
 * 
 * Handles collection name, file selection, metadata schema, and upload state
 * throughout the multi-step collection creation workflow.
 * 
 * @returns Store with collection creation state and actions
 * 
 * @example
 * ```tsx
 * const { collectionName, selectedFiles, setCollectionName, addFiles } = useNewCollectionStore();
 * setCollectionName("my-collection");
 * addFiles([file1, file2]);
 * ```
 */
export const useNewCollectionStore = create<NewCollectionState>((set, get) => ({
  collectionName: "",
  collectionNameTouched: false,
  selectedFiles: [],
  fileMetadata: {},
  metadataSchema: [],
  isLoading: false,
  uploadComplete: false,
  error: null,
  hasInvalidFiles: false,

  setCollectionName: (name) => set({ collectionName: name }),
  setCollectionNameTouched: (touched) => set({ collectionNameTouched: touched }),
  setIsLoading: (v) => set({ isLoading: v }),
  setUploadComplete: (v) => set({ uploadComplete: v }),
  setError: (msg) => set({ error: msg }),
  setHasInvalidFiles: (hasInvalidFiles) => set({ hasInvalidFiles }),

  setMetadataSchema: (schema) => {
    const { selectedFiles, fileMetadata } = get();
    const updatedMetadata: Record<string, Record<string, unknown>> = {};

    for (const file of selectedFiles) {
      const existing = fileMetadata[file.name] || {};
      updatedMetadata[file.name] = {};

      for (const field of schema) {
        updatedMetadata[file.name][field.name] = existing[field.name] ?? getDefaultValueForField(field);
      }
    }

    set({
      metadataSchema: schema,
      fileMetadata: updatedMetadata,
    });
  },

  addFiles: (files) => {
    const { selectedFiles, metadataSchema, fileMetadata } = get();
    const updated = [...selectedFiles, ...files];
    const updatedMetadata = { ...fileMetadata };

    for (const file of files) {
      if (!updatedMetadata[file.name]) {
        updatedMetadata[file.name] = {};
        for (const field of metadataSchema) {
          updatedMetadata[file.name][field.name] = getDefaultValueForField(field);
        }
      }
    }

    set({
      selectedFiles: updated,
      fileMetadata: updatedMetadata,
    });
  },

  setFiles: (files) => {
    console.log('🟢 Store: setFiles called with', files.length, 'files');
    const { metadataSchema } = get();
    const fileMetadata: Record<string, Record<string, unknown>> = {};

    for (const file of files) {
      fileMetadata[file.name] = {};
      for (const field of metadataSchema) {
        fileMetadata[file.name][field.name] = getDefaultValueForField(field);
      }
    }

    set({
      selectedFiles: files,
      fileMetadata,
    });
    console.log('🟢 Store: selectedFiles is now', get().selectedFiles.length, 'files');
  },

  removeFile: (index) => {
    const { selectedFiles, fileMetadata } = get();
    const file = selectedFiles[index];
    const updatedFiles = selectedFiles.filter((_, i) => i !== index);
    const updatedMetadata = { ...fileMetadata };
    delete updatedMetadata[file.name];

    set({
      selectedFiles: updatedFiles,
      fileMetadata: updatedMetadata,
    });
  },

  updateMetadataField: (filename, field, value) => {
    const { fileMetadata, metadataSchema } = get();
    
    // Find the field definition to ensure proper typing
    const fieldDef = metadataSchema.find(f => f.name === field);
    let processedValue = value;
    
    // Type conversion based on field definition
    if (fieldDef) {
      switch (fieldDef.type) {
        case "integer":
          if (typeof value === "string") {
            const num = parseInt(value);
            processedValue = isNaN(num) || value.trim() === "" ? null : num;
          }
          break;
        case "float":
        case "number":
          if (typeof value === "string") {
            const num = parseFloat(value);
            processedValue = isNaN(num) || value.trim() === "" ? null : num;
          }
          break;
        case "boolean":
          if (typeof value === "string") {
            processedValue = value === "true";
          }
          break;
        case "array":
          // Arrays should already be processed as arrays or JSON strings
          if (typeof value === "string" && value.startsWith("[")) {
            try {
              processedValue = JSON.parse(value);
            } catch {
              processedValue = [];
            }
          }
          break;
        case "string":
        case "datetime":
        default:
          // Keep as-is for strings and datetime
          break;
      }
    }
    
    set({
      fileMetadata: {
        ...fileMetadata,
        [filename]: {
          ...fileMetadata[filename],
          [field]: processedValue,
        },
      },
    });
  },

  reset: () =>
    set({
      collectionName: "",
      collectionNameTouched: false,
      selectedFiles: [],
      fileMetadata: {},
      metadataSchema: [],
      isLoading: false,
      uploadComplete: false,
      error: null,
      hasInvalidFiles: false,
    }),
}));

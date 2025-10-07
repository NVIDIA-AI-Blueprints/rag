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

// src/hooks/useNewCollectionForm.ts

import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useCollections, useCreateCollection } from "../api/useCollectionsApi";
import { useCollectionsStore } from "../store/useCollectionsStore";
import { useSettingsStore } from "../store/useSettingsStore";
import type { Collection, UIMetadataField } from "../types/collections";
import type { CreateCollectionPayload } from "../types/api";

export function useNewCollectionForm() {
  const navigate = useNavigate();
  const { data: existingCollections = [] } = useCollections();
  const { vdbEndpoint } = useSettingsStore();
  const { mutateAsync: createCollection } = useCreateCollection();
  const refresh = useCollectionsStore((s) => s.refresh);

  const [collectionName, setCollectionName] = useState("");
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);

  const [fileMetadata, setFileMetadata] = useState<Record<string, Record<string, unknown>>>({});
  const [metadataSchema, setMetadataSchema] = useState<UIMetadataField[]>([]);
  const [uploadComplete, setUploadComplete] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileSelect = (files: File[]) => {
    const updated = [...selectedFiles, ...files];
    const newMetadata = { ...fileMetadata };
    for (const file of files) {
      if (!newMetadata[file.name]) {
        newMetadata[file.name] = {};
        for (const field of metadataSchema) {
          newMetadata[file.name][field.name] = "";
        }
      }
    }
    setSelectedFiles(updated);
    setFileMetadata(newMetadata);
  };

  const removeFile = (index: number) => {
    const file = selectedFiles[index];
    setSelectedFiles((prev) => prev.filter((_, i) => i !== index));
    setFileMetadata((prev) => {
      const copy = { ...prev };
      delete copy[file.name];
      return copy;
    });
  };

  const handleMetadataChange = (filename: string, field: string, value: unknown) => {
    setFileMetadata((prev) => ({
      ...prev,
      [filename]: { ...prev[filename], [field]: value },
    }));
  };

  const handleCollectionNameChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setCollectionName(e.target.value.replace(/\s+/g, "_"));
  };

  const hasMissingRequired = selectedFiles.some((file) =>
    metadataSchema.some(
      (field) => {
        if (field.optional) return false;
        const value = fileMetadata[file.name]?.[field.name];
        if (field.type === "string") {
          return !value || (typeof value === "string" && !value.trim());
        }
        return value === undefined || value === null;
      }
    )
  );

  const handleSubmit = async () => {
    try {
      const exists = existingCollections.some((c: Collection) => c.collection_name === collectionName);
      if (exists) {
        setError("A collection with this name already exists.");
        return;
      }

      setIsLoading(true);
      setError(null);

      const collectionPayload: CreateCollectionPayload = {
        collection_name: collectionName,
        embedding_dimension: 2048,
        metadata_schema: metadataSchema.map(({ name, type }) => ({
          name: name.trim(),
          type,
          description: `${name.trim()} field`,
        })),
      };

      // Only include vdb_endpoint if explicitly set by user
      if (vdbEndpoint) {
        collectionPayload.vdb_endpoint = vdbEndpoint;
      }

      await createCollection(collectionPayload);

      if (selectedFiles.length > 0) {
        const formData = new FormData();
        selectedFiles.forEach((file) => formData.append("documents", file));

        formData.append(
          "data",
          JSON.stringify({
            collection_name: collectionName,
            blocking: false,
            custom_metadata: selectedFiles.map((file) => ({
              filename: file.name,
              metadata: fileMetadata[file.name] || {},
            })),
          })
        );

        const uploadResp = await fetch(`/api/documents`, {
          method: "POST",
          body: formData,
        });

        if (!uploadResp.ok) throw new Error("Failed to upload documents");
      }

      refresh();
      setUploadComplete(true);
      setSelectedFiles([]);
    } catch (err) {
      console.error(err);
      setError(err instanceof Error ? err.message : "Unexpected error");
    } finally {
      setIsLoading(false);
    }
  };

  return {
    collectionName,
    selectedFiles,
    fileMetadata,
    metadataSchema,
    setMetadataSchema,
    uploadComplete,
    isLoading,
    error,
    handleFileSelect,
    removeFile,
    handleMetadataChange,
    handleCollectionNameChange,
    handleSubmit,
    hasMissingRequired,
    navigate,
  };
}

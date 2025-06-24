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
"use client";

import { useState, useEffect } from "react";
import { useApp } from "@/app/context/AppContext";
import Modal from "../Modal/Modal";
import MetadataSchemaEditor from "../RightSidebar/MetadataSchemaEditor";
import { UIMetadataField } from "@/types/collections";

interface NewCollectionModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess?: () => void;
}

export default function NewCollectionModal({
  isOpen,
  onClose,
  onSuccess
}: NewCollectionModalProps) {
  const { setCollections, addPendingTask } = useApp();
  const [collectionName, setCollectionName] = useState("");
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [fileMetadata, setFileMetadata] = useState<Record<string, Record<string, string>>>({});
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploadComplete, setUploadComplete] = useState(false);
  const [showFileInput, setShowFileInput] = useState(true);
  const [metadataSchema, setMetadataSchema] = useState<UIMetadataField[]>([]);

  useEffect(() => {
    if (isOpen) {
      setCollectionName("");
      setSelectedFiles([]);
      setFileMetadata({});
      setError(null);
      setUploadComplete(false);
      setShowFileInput(true);
      setMetadataSchema([]);
    }
  }, [isOpen]);

  const handleFileSelect = (files: File[]) => {
    setSelectedFiles((prev) => {
      const updated = [...prev, ...files];

      const newMetadata = { ...fileMetadata };
      for (const file of files) {
        if (!newMetadata[file.name]) {
          newMetadata[file.name] = {};
          for (const field of metadataSchema) {
            newMetadata[file.name][field.name] = "";
          }
        }
      }

      setFileMetadata(newMetadata);
      return updated;
    });
  };

  const removeFile = (index: number) => {
    const removedFile = selectedFiles[index];
    setSelectedFiles((prev) => prev.filter((_, i) => i !== index));
    setFileMetadata((prev) => {
      const updated = { ...prev };
      delete updated[removedFile.name];
      return updated;
    });
  };

  const handleMetadataChange = (filename: string, field: string, value: string) => {
    setFileMetadata(prev => ({
      ...prev,
      [filename]: {
        ...prev[filename],
        [field]: value
      }
    }));
  };

  const handleCollectionNameChange = (
    e: React.ChangeEvent<HTMLInputElement>
  ) => {
    setCollectionName(e.target.value.replace(/\s+/g, "_"));
  };

  const handleReset = () => {
    setCollectionName("");
    setSelectedFiles([]);
    setFileMetadata({});
    setError(null);
  };

  const handleSubmit = async () => {
    try {
      if (!collectionName.match(/^[a-zA-Z_][a-zA-Z0-9_]*$/)) {
        setError("Collection name must start with a letter or underscore and can only contain letters, numbers and underscores");
        return;
      }

      const checkResponse = await fetch("/api/collections");
      if (!checkResponse.ok) throw new Error("Failed to check existing collections");

      const { collections: existingCollections } = await checkResponse.json();
      if (existingCollections.some((c: any) => c.collection_name === collectionName)) {
        setError("A collection with this name already exists");
        return;
      }

      const cleanedSchema = metadataSchema.map((field) => ({
        name: field.name,
        type: field.type,
      }));

      setIsLoading(true);
      setError(null);

      const createCollectionResponse = await fetch("/api/collection", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          collection_name: collectionName,
          embedding_dimension: 2048,
          metadata_schema: cleanedSchema,
        }),
      });

      if (!createCollectionResponse.ok) throw new Error("Failed to create collection");

      const collectionData = await createCollectionResponse.json();
      if (collectionData.failed?.length > 0) {
        throw new Error(
          `Failed to create collection: ${collectionData.message || "Unknown error"}`
        );
      }

      if (selectedFiles.length > 0) {
        const formData = new FormData();
        selectedFiles.forEach((file) => {
          formData.append("documents", file);
        });

        const metadata = {
          collection_name: collectionName,
          blocking: false,
          custom_metadata: selectedFiles.map((file) => ({
            filename: file.name,
            metadata: fileMetadata[file.name] || {}
          }))
        };

        formData.append("data", JSON.stringify(metadata));

        const uploadResponse = await fetch("/api/documents?blocking=false", {
          method: "POST",
          body: formData,
        });

        if (!uploadResponse.ok) {
          throw new Error("Failed to upload documents");
        }

        const uploadData = await uploadResponse.json();

        if (uploadData.task_id) {
          const documentNames = selectedFiles.map(file => file.name);
          addPendingTask({
            id: uploadData.task_id,
            collection_name: collectionName,
            state: "PENDING",
            created_at: new Date().toISOString(),
            documents: documentNames
          });
        }
      }

      const getCollectionsResponse = await fetch("/api/collections");
      if (!getCollectionsResponse.ok) {
        throw new Error("Failed to fetch updated collections");
      }

      const { collections } = await getCollectionsResponse.json();
      setCollections(
        collections.map((collection: any) => ({
          collection_name: collection.collection_name,
          document_count: collection.num_entities,
          index_count: collection.num_entities,
          metadata_schema: collection.metadata_schema ?? [],
        }))
      );

      setUploadComplete(true);
      setShowFileInput(false);
      setSelectedFiles([]);
      setIsLoading(false);
      onSuccess?.();
    } catch (err) {
      console.error("Error creating collection:", err);
      setError(err instanceof Error ? err.message : "An error occurred");
      setIsLoading(false);
    }
  };

  const modalDescription =
    "Upload a collection of source files to provide the model with relevant information for more tailored responses (e.g., marketing plans, research notes, meeting transcripts, sales documents).";

  const collectionNameInput = (
    <div className="mb-4">
      <label className="mb-2 block text-sm font-medium">Collection Name</label>
      <input
        type="text"
        value={collectionName}
        onChange={handleCollectionNameChange}
        placeholder="Enter collection name"
        className="w-full rounded-md bg-neutral-800 px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-[var(--nv-green)]"
        disabled={isLoading || uploadComplete}
      />
    </div>
  );

  const renderSuccessMessage = () => {
    if (!uploadComplete) return null;

    return (
      <div className="mb-4 mt-2 overflow-hidden rounded-md border border-neutral-700 bg-neutral-900 p-3">
        <div className="mb-2 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
              <path d="M6.00016 10.7799L3.22016 7.99987L2.27349 8.93987L6.00016 12.6665L14.0002 4.66654L13.0602 3.72654L6.00016 10.7799Z" fill="#22C55E"/>
            </svg>
            <span className="font-medium text-sm">Collection created successfully</span>
          </div>
          <button
            onClick={() => setUploadComplete(false)}
            className="text-neutral-500 hover:text-white transition-colors"
            title="Dismiss"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
              <path d="M18 6L6 18M6 6L18 18" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </button>
        </div>
        <div className="mt-2 text-xs">
          <p className="text-neutral-400">
            Your new collection "{collectionName}" has been created.
            {selectedFiles.length > 0 && (
              <> Documents are being processed and will be available soon. You can view processing status in the Add Source dialog.</>
            )}
          </p>
        </div>
      </div>
    );
  };

  const hasMissingRequired = selectedFiles.some(file =>
    metadataSchema.some(field =>
      !field.optional && !fileMetadata[file.name]?.[field.name]?.trim()
    )
  );

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title="New Collection"
      description={modalDescription}
      isLoading={isLoading}
      error={error}
      selectedFiles={selectedFiles}
      submitButtonText="Create Collection"
      isSubmitDisabled={hasMissingRequired || !collectionName}
      onFileSelect={handleFileSelect}
      onRemoveFile={removeFile}
      onReset={handleReset}
      onSubmit={handleSubmit}
      fileInputId="fileInput"
      fileMetadata={fileMetadata}
      onMetadataChange={handleMetadataChange}
      metadataSchema={metadataSchema}
      customContent={
        <>
          {collectionNameInput}
          <MetadataSchemaEditor schema={metadataSchema} setSchema={setMetadataSchema} />
          {renderSuccessMessage()}
        </>
      }
      showFileInput={showFileInput && !uploadComplete}
      hideActionButtons={uploadComplete}
    />
  );
}
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

import { useState, useEffect, useCallback, useMemo } from "react";
import Image from "next/image";
import NewCollectionModal from "./NewCollectionModal";
import CollectionItem from "./CollectionItem";
import SourceItem from "./SourceItem";
import AddSourceModal from "./AddSourceModal";
import { useApp } from "../../context/AppContext";
import { CollectionResponse } from "@/types/collections";
import { DocumentResponse } from "@/types/documents";
import React from "react";

export default function Collections() {
  // State and Context
  const {
    collections,
    selectedCollections,
    setSelectedCollections,
    setCollections,
    onDocumentsUpdated,
  } = useApp();
  const selectedCollection = selectedCollections[0] || null;

  const [searchQuery, setSearchQuery] = useState("");
  const [isNewCollectionModalOpen, setIsNewCollectionModalOpen] =
    useState(false);
  const [isAddSourceModalOpen, setIsAddSourceModalOpen] = useState(false);
  const [showSourceItems, setShowSourceItems] = useState(false);
  const [sourceItems, setSourceItems] = useState<DocumentResponse[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isLoadingCollections, setIsLoadingCollections] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [modalCollectionName, setModalCollectionName] = useState("");

  // Data Fetching
  useEffect(() => {
    fetchCollections();
  }, [setCollections]);

  // Define fetchDocuments with useCallback to avoid recreating it in each render
  const fetchDocuments = useCallback(async () => {
    if (!selectedCollection) return;

    try {
      // Only set loading to true if we don't already have documents loaded
      // This prevents the flickering when refreshing existing documents
      if (sourceItems.length === 0) {
        setIsLoading(true);
      }

      setError(null);
      const response = await fetch(
        `/api/documents?collection_name=${selectedCollection}`
      );
      if (!response.ok) throw new Error("Failed to fetch documents");

      const data = await response.json();

      // Only update state if the documents have actually changed
      if (JSON.stringify(data.documents) !== JSON.stringify(sourceItems)) {
        setSourceItems(data.documents);
      }
    } catch (error) {
      console.error("Error fetching documents:", error);
      setError("Failed to load documents");
    } finally {
      setIsLoading(false);
    }
  }, [selectedCollection, sourceItems]);

  useEffect(() => {
    if (selectedCollection && showSourceItems) {
      // Use a slight delay to ensure state updates have been processed
      const timer = setTimeout(() => {
        fetchDocuments();
      }, 0);

      return () => clearTimeout(timer);
    }
  }, [selectedCollection, showSourceItems, fetchDocuments]);

  // Subscribe to document updates for the selected collection
  useEffect(() => {
    if (!selectedCollection) return;

    // Set up the callback for updating documents when tasks complete
    const unsubscribe = onDocumentsUpdated(selectedCollection, () => {
      // Only fetch documents if we're currently viewing them
      if (showSourceItems) {
        fetchDocuments();
      }
    });

    // Clean up the subscription when component unmounts or selectedCollection changes
    return unsubscribe;
  }, [selectedCollection, showSourceItems, onDocumentsUpdated, fetchDocuments]);

  // API Calls
  const fetchCollections = async () => {
    try {
      setIsLoadingCollections(true);
      setError(null);
      const response = await fetch("/api/collections");
      if (!response.ok) throw new Error("Failed to fetch collections");

      const data = await response.json();
      setCollections(
        data.collections.map((collection: CollectionResponse) => ({
          collection_name: collection.collection_name,
          document_count: collection.num_entities,
          index_count: collection.num_entities,
          metadata_schema: collection.metadata_schema,
        }))
      );
    } catch (error) {
      console.error("Error fetching collections:", error);
      setError("Failed to load collections");
    } finally {
      setIsLoadingCollections(false);
    }
  };

  // Event Handlers
  const handleDeleteCollection = async (name: string) => {
    try {
      const response = await fetch("/api/collections", {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ collection_names: [name] }),
      });

      if (!response.ok) throw new Error("Failed to delete collection");

      setCollections(collections.filter((c) => c.collection_name !== name));
      setSelectedCollections((prev) => prev.filter((c) => c !== name));
      if (selectedCollection === name) {
        setShowSourceItems(false);
      }
    } catch (error) {
      console.error("Error deleting collection:", error);
      setError("Failed to delete collection");
    }
  };

  const handleDeleteDocument = async (documentName: string) => {
    if (!selectedCollection) return;

    try {
      const response = await fetch(
        `/api/documents?collection_name=${selectedCollection}`,
        {
          method: "DELETE",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify([documentName]),
        }
      );

      if (!response.ok) throw new Error("Failed to delete document");

      await response.json();

      await fetchDocuments();

      const collectionsResponse = await fetch("/api/collections");
      if (collectionsResponse.ok) {
        const { collections: updatedCollections } =
          await collectionsResponse.json();
        setCollections(
          updatedCollections.map((collection: any) => ({
            collection_name: collection.collection_name,
            document_count: collection.num_entities,
            index_count: collection.num_entities,
          }))
        );
      }
    } catch (error) {
      console.error("Error deleting document:", error);
      setError("Failed to delete document");
      await fetchDocuments();
    }
  };

  const handleViewFiles = (collectionName: string) => {
    // Use React 18's automatic batching or wrap in a function to batch these updates
    React.startTransition(() => {
      setSelectedCollections([collectionName]);
      setShowSourceItems(true);
    });
  };

  const handleCollectionSelect = (collectionName: string) => {
    setSelectedCollections((prev) =>
      prev.includes(collectionName)
        ? prev.filter((name) => name !== collectionName)
        : [...prev, collectionName]
    );
    setShowSourceItems(false);
  };

  const handleBackToCollections = () => {
    setShowSourceItems(false);
    setSourceItems([]);
  };

  // Function to open AddSourceModal for a specific collection
  const openAddSourceModal = (collectionName: string) => {
    setModalCollectionName(collectionName);
    setIsAddSourceModalOpen(true);
  };

  // Render Helpers
  const renderHeader = () => (
    <div className="mb-4 flex items-center justify-between">
      <div className="flex items-center gap-2">
        <h2 className="text-sm font-medium">
          {showSourceItems ? (
            <button
              onClick={handleBackToCollections}
              className="whitespace-nowrap hover:text-[var(--nv-green)]"
            >
              All Collections
            </button>
          ) : (
            <span className="whitespace-nowrap">All Collections</span>
          )}
        </h2>
        {showSourceItems && selectedCollection && (
          <>
            <span className="text-gray-500">/</span>
            <span
              title={selectedCollection}
              className="inline-block max-w-[170px] truncate text-sm font-medium text-[var(--nv-green)]"
            >
              {selectedCollection}
            </span>
          </>
        )}
      </div>
    </div>
  );

  const renderContent = () => {
    if (isLoadingCollections) {
      return <LoadingState message="Loading collections..." />;
    }

    if (error) {
      return <ErrorState error={error} />;
    }

    if (collections.length === 0) {
      return <EmptyState />;
    }

    return (
      <>
        <div className="collections-container relative">
          <div className="max-h-[calc(100vh-260px)] overflow-y-auto pr-2">
            {showSourceItems ? <DocumentsList /> : <CollectionsList />}
          </div>
        </div>
      </>
    );
  };

  // Render Components
  const LoadingState = ({ message }: { message: string }) => (
    <div className="flex h-[200px] items-center justify-center">
      <div className="flex flex-col items-center gap-4">
        <div className="h-6 w-6 animate-spin rounded-full border-2 border-[var(--nv-green)] border-t-transparent" />
        <p className="text-sm text-gray-400">{message}</p>
      </div>
    </div>
  );

  const ErrorState = ({ error }: { error: string }) => (
    <div className="flex h-[200px] flex-col items-center justify-center text-center">
      <div className="mb-4">
        <Image
          src="/error.svg"
          alt="Error"
          width={48}
          height={48}
          className="opacity-50"
        />
      </div>
      <p className="mb-2 text-sm text-red-400">{error}</p>
      <button
        onClick={() => window.location.reload()}
        className="text-xs text-[var(--nv-green)] hover:underline"
      >
        Try again
      </button>
    </div>
  );

  const EmptyState = () => (
    <div className="flex h-[200px] flex-col items-center justify-center text-center">
      <div className="mb-4">
        <Image
          src="/empty-collections.svg"
          alt="No collections"
          width={48}
          height={48}
          className="opacity-50"
        />
      </div>
      <p className="mb-2 text-sm text-gray-400">No collections</p>
      <p className="text-xs text-gray-500">
        Create your first collection and add files to customize your model
        response.
      </p>
    </div>
  );

  // Memoize the rendered components to prevent unnecessary re-renders
  const DocumentsList = useMemo(() => {
    return () => (
      <div className="documents-container relative">
        <div className="max-h-[calc(100vh-260px)] space-y-1 overflow-y-auto pr-2">
          {isLoading ? (
            <LoadingState message="Loading documents..." />
          ) : sourceItems.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-8 text-center">
              <Image
                src="/document.svg"
                alt="No documents"
                width={48}
                height={48}
                className="mb-4 opacity-50"
              />
              <p className="text-sm text-gray-400">
                No documents in this collection
              </p>
            </div>
          ) : (
            sourceItems.map((item) => (
              console.log("item.metadata", item),
              <SourceItem
                key={item.document_name}
                name={item.document_name}
                metadata={item.metadata}
                onDelete={() => handleDeleteDocument(item.document_name)}
              />
            ))
          )}
          {error && (
            <div className="mt-2 rounded-md bg-red-900/50 p-2 text-sm text-red-200">
              {error}
            </div>
          )}
        </div>
      </div>
    );
  }, [sourceItems, isLoading, error, handleDeleteDocument]);

  const CollectionsList = useMemo(() => {
    return () => (
      <>
        {collections
          .filter((collection) =>
            collection.collection_name
              .toLowerCase()
              .includes(searchQuery.toLowerCase())
          )
          .map((collection) => (
            <CollectionItem
              key={collection.collection_name}
              name={collection.collection_name}
              metadataSchema={collection.metadata_schema}
              isSelected={selectedCollections.includes(collection.collection_name)}
              onSelect={() => handleCollectionSelect(collection.collection_name)}
              onDelete={() => handleDeleteCollection(collection.collection_name)}
              handleViewFiles={handleViewFiles}
              onDocumentsUpdate={fetchDocuments}
              onShowTaskStatus={() => openAddSourceModal(collection.collection_name)}
            />
          ))}
      </>
    );
  }, [
    collections,
    searchQuery,
    selectedCollections,
    handleCollectionSelect,
    handleDeleteCollection,
    handleViewFiles,
    fetchDocuments,
    openAddSourceModal,
  ]);

  return (
    <div className="flex w-[320px] flex-col bg-black p-4 text-white">
      <div className="relative mb-6">
        <input
          type="text"
          placeholder="Search collections"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="w-full rounded-md bg-neutral-900 px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-[var(--nv-green)]"
          disabled={isLoadingCollections}
        />
      </div>

      <div className="flex-1">
        {renderHeader()}
        {renderContent()}
      </div>

      <div className="mt-auto flex gap-2">
        <button
          className="flex w-full items-center justify-center gap-2 rounded-full border border-[var(--nv-green)] bg-black px-4 py-2 font-medium text-white transition-colors hover:bg-[#1A1A1A] disabled:cursor-not-allowed disabled:opacity-50"
          onClick={() => setIsNewCollectionModalOpen(true)}
          disabled={isLoadingCollections}
        >
          <span className="text-sm">New Collection</span>
        </button>
        <button
          className="flex w-full items-center justify-center gap-2 rounded-full border border-[var(--nv-green)] bg-black px-4 py-2 font-medium text-white transition-colors hover:bg-[#1A1A1A] disabled:cursor-not-allowed disabled:opacity-50"
          disabled={
            collections.length === 0 ||
            isLoadingCollections ||
            (showSourceItems && !selectedCollection)
          }
          onClick={() => {
            setModalCollectionName(selectedCollection || "");
            setIsAddSourceModalOpen(true);
          }}
        >
          <span className="text-sm">Add Source</span>
        </button>

        <NewCollectionModal
          isOpen={isNewCollectionModalOpen}
          onSuccess={() => {
            fetchCollections(); // ← trigger immediate reload
          }}
          onClose={() => setIsNewCollectionModalOpen(false)}
        />
        <AddSourceModal
          key={modalCollectionName || selectedCollection || ""}
          isOpen={isAddSourceModalOpen}
          onClose={() => setIsAddSourceModalOpen(false)}
          collectionName={modalCollectionName || selectedCollection || ""}
          onDocumentsUpdate={fetchDocuments}
        />
      </div>
    </div>
  );
}

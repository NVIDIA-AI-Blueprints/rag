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
import { useApp } from "../../context/AppContext";
import Modal from "../Modal/Modal";
import { UIMetadataField } from "@/types/collections";
import MetadataSchemaEditor from "../RightSidebar/MetadataSchemaEditor";

interface AddSourceModalProps {
  isOpen: boolean;
  onClose: () => void;
  collectionName: string;
  onDocumentsUpdate: () => void;
}

interface SuccessfulDocument {
  document_id: string;
  document_name: string;
  size_bytes?: number;
}

interface FailedDocument {
  document_name: string;
  error_message?: string;
}

interface TaskResult {
  message: string;
  total_documents: number;
  documents: SuccessfulDocument[];
  failed_documents: FailedDocument[];
}

export default function AddSourceModal({
  isOpen,
  onClose,
  collectionName,
  onDocumentsUpdate,
}: AddSourceModalProps) {
  const { collections, setCollections, addPendingTask, pendingTasks, removePendingTask } = useApp();
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [fileMetadata, setFileMetadata] = useState<Record<string, Record<string, string>>>({});
  const [selectedCollection, setSelectedCollection] = useState(collectionName);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showDocumentUpload, setShowDocumentUpload] = useState(true);
  const [metadataSchema, setMetadataSchema] = useState<UIMetadataField[]>([]);

  const currentCollection = collections.find(c => c.collection_name === selectedCollection);
  
  // Add state to track which file lists are expanded
  const [expandedLists, setExpandedLists] = useState<{[key: string]: boolean}>({});
  
  // Get all tasks for the current collection
  const collectionTasks = pendingTasks.filter(
    task => task.collection_name === selectedCollection
  ).sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());

  useEffect(() => {
    setSelectedCollection(collectionName);
  }, [collectionName]);

  useEffect(() => {
    if (isOpen && currentCollection?.metadata_schema) {
      setMetadataSchema(currentCollection.metadata_schema);
    }
  }, [isOpen, collectionName, selectedCollection]);

  // Reset state when modal opens/closes
  useEffect(() => {
    if (isOpen) {
      setSelectedFiles([]);
      setError(null);
      setShowDocumentUpload(true);
      setExpandedLists({}); // Reset expanded lists when modal opens
    }
  }, [isOpen]);

  // Toggle expanded state for a specific list
  const toggleExpanded = (listId: string) => {
    setExpandedLists(prev => ({
      ...prev,
      [listId]: !prev[listId]
    }));
  };

    // Update fileMetadata when files change (reset or add)
  useEffect(() => {
    // Initialize metadata for new files if not present
    const newMetadata = { ...fileMetadata };
    selectedFiles.forEach((file) => {
      if (!newMetadata[file.name]) {
        newMetadata[file.name] = {};
      }
    });
    setFileMetadata(newMetadata);
  }, [selectedFiles]);

  // Function to fetch existing documents and check for duplicates
  const checkForDuplicates = async (files: File[], collection: string) => {
    try {
      const response = await fetch(
        `/api/documents?collection_name=${encodeURIComponent(collection)}`
      );

      if (!response.ok) {
        if (response.status === 400) {
          console.error("Bad request when checking for duplicates");
          return [];
        }
        throw new Error(`Error fetching documents: ${response.status}`);
      }

      const data = await response.json();

      // Extract document_name values from documents
      const existingFiles = data.documents
        .map((doc: any) => doc.document_name || null)
        .filter(Boolean);

      // Find duplicates by comparing file names with document_name values
      const duplicates = files.filter((file) =>
        existingFiles.includes(file.name)
      );

      return duplicates;
    } catch (err) {
      console.error("Error checking for duplicate files:", err);
      return [];
    }
  };

  const handleFileSelect = (files: File[]) => {
    setSelectedFiles((prev) => [...prev, ...files]);
    setError(null);
  };

  const removeFile = (index: number) => {
    setSelectedFiles((prev) => prev.filter((_, i) => i !== index));
    setFileMetadata((prev) => {
      const updated = { ...prev };
      delete updated[removeFile.name];
      return updated;
    });
    setError(null);
  };

  const handleReset = () => {
    setSelectedCollection(collectionName);
    setSelectedFiles([]);
    setError(null);
  };

  const dismissTask = (taskId: string) => {
    if (removePendingTask) {
      removePendingTask(taskId);
    }
  };

  const handleMetadataChange = (fileName: string, fieldName: string, value: string) => {
    setFileMetadata((prev) => ({
      ...prev,
      [fileName]: {
        ...prev[fileName],
        [fieldName]: value,
      },
    }));
  };

  const handleSubmit = async () => {
    try {
      // 1. Check for duplicate files in the collection
      const duplicates = await checkForDuplicates(
        selectedFiles,
        selectedCollection
      );

      if (duplicates.length > 0) {
        const duplicateNames = duplicates.map((file) => file.name).join(", ");
        setError(
          `The following files already exist in this collection: ${duplicateNames}`
        );
        return;
      }

      setIsLoading(true);
      setError(null);

      // 2. Prepare and upload documents to the collection
      const formData = new FormData();
      selectedFiles.forEach((file) => {
        formData.append("documents", file);
      });

      // Add metadata as JSON string with blocking=false
      const metadata = {
        collection_name: selectedCollection,
        blocking: false,
        custom_metadata: selectedFiles.map(file => {
          const rawMetadata = fileMetadata[file.name] || {};
          const cleaned = Object.fromEntries(
            Object.entries(rawMetadata).filter(([_, v]) => v != null && v.trim() !== "")
          );
          return {
            filename: file.name,
            ...(Object.keys(cleaned).length > 0 ? { metadata: cleaned } : {}),
          };
        }),
      };
      formData.append("data", JSON.stringify(metadata));

      // Use URL parameter as backup for blocking=false
      const uploadUrl = "/api/documents?blocking=false";
      const uploadResponse = await fetch(uploadUrl, {
        method: "POST",
        body: formData,
      });

      if (!uploadResponse.ok) {
        throw new Error("Failed to upload documents");
      }

      // Get task information from response
      const responseData = await uploadResponse.json();
      
      // Add task to pending tasks if we got a task_id
      if (responseData.task_id) {
        const documentNames = selectedFiles.map(file => file.name);
        
        addPendingTask({
          id: responseData.task_id,
          collection_name: selectedCollection,
          state: "PENDING",
          created_at: new Date().toISOString(),
          documents: documentNames
        });
        
        console.log(`Added task ${responseData.task_id} to pending tasks`);
      }

      // 3. Update the collections list in the UI
      const getCollectionsResponse = await fetch("/api/collections");
      if (!getCollectionsResponse.ok) {
        throw new Error("Failed to fetch updated collections");
      }

      const { collections: updatedCollections } =
        await getCollectionsResponse.json();
      setCollections(
        updatedCollections.map((collection: any) => ({
          collection_name: collection.collection_name,
          document_count: collection.num_entities,
          index_count: collection.num_entities,
          metadata_schema: collection.metadata_schema ?? [],
        }))
      );

      // 4. Trigger document list refresh and update UI
      if (onDocumentsUpdate) {
        onDocumentsUpdate();
      }

      // Clear selected files after upload
      setSelectedFiles([]);
      setIsLoading(false);
    } catch (err) {
      console.error("Error uploading documents:", err);
      setError(err instanceof Error ? err.message : "An error occurred");
      setIsLoading(false);
    }
  };

  const modalDescription =
    "Upload a collection of source files to provide the model with relevant information for more tailored responses (e.g., marketing plans, research notes, meeting transcripts, sales documents).";

  const collectionSelector = (
    <div className="mb-4">
      <label className="mb-2 block text-sm font-medium">Collection</label>
      <select
        value={selectedCollection}
        onChange={(e) => {
          setSelectedCollection(e.target.value);
        }}
        className="w-full rounded-md bg-neutral-800 px-4 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-[var(--nv-green)]"
        disabled={isLoading}
      >
        {collections.map((collection) => (
          <option
            key={collection.collection_name}
            value={collection.collection_name}
          >
            {collection.collection_name}
          </option>
        ))}
      </select>
    </div>
  );

  // Render all tasks for this collection
  const renderTaskStatuses = () => {
    if (collectionTasks.length === 0) return null;
    
    return (
      <div className="mb-6 mt-3 space-y-3">
        <h3 className="text-sm font-medium">Processing Status</h3>
        <div className="max-h-[min(300px,40vh)] overflow-y-auto pr-1 custom-scrollbar">
          {collectionTasks.map(task => (
            <div key={task.id} className="mb-3 overflow-hidden rounded-md border border-neutral-700 bg-neutral-900 p-3">
              {/* Status Header */}
              <div className="mb-3 flex items-center justify-between">
                <div className="flex items-center gap-2 overflow-hidden">
                  {task.state === "PENDING" ? (
                    <div className="flex h-5 w-5 flex-shrink-0 items-center justify-center bg-yellow-500/10 rounded-full">
                      <div className="h-2 w-2 animate-pulse rounded-full bg-yellow-400"></div>
                    </div>
                  ) : task.state === "FINISHED" && (!task.result?.failed_documents || task.result.failed_documents.length === 0) ? (
                    <div className="flex h-5 w-5 flex-shrink-0 items-center justify-center bg-green-500/10 rounded-full">
                      <svg width="14" height="14" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M6.00016 10.7799L3.22016 7.99987L2.27349 8.93987L6.00016 12.6665L14.0002 4.66654L13.0602 3.72654L6.00016 10.7799Z" fill="#22C55E"/>
                      </svg>
                    </div>
                  ) : task.state === "FINISHED" && task.result?.failed_documents && task.result.failed_documents.length > 0 ? (
                    <div className="flex h-5 w-5 flex-shrink-0 items-center justify-center bg-yellow-500/10 rounded-full">
                      <svg width="14" height="14" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M7.99992 3.99998H7.99992V10H7.99992V3.99998ZM7.99992 12V12V14H7.99992V12Z" stroke="#F59E0B" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                      </svg>
                    </div>
                  ) : (
                    <div className="flex h-5 w-5 flex-shrink-0 items-center justify-center bg-red-500/10 rounded-full">
                      <svg width="14" height="14" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M8.00016 6.94654L4.97349 3.91987L3.91349 4.97987L6.94016 8.00654L3.91349 11.0332L4.97349 12.0932L8.00016 9.06654L11.0268 12.0932L12.0868 11.0332L9.06016 8.00654L12.0868 4.97987L11.0268 3.91987L8.00016 6.94654Z" fill="#EF4444"/>
                      </svg>
                    </div>
                  )}
                  <div className="overflow-hidden">
                    <span className={`font-medium text-sm ${
                      task.state === "PENDING" ? "text-yellow-400" : 
                      task.state === "FINISHED" && (!task.result?.failed_documents || task.result.failed_documents.length === 0) ? "text-green-500" : 
                      task.state === "FINISHED" && task.result?.failed_documents && task.result.failed_documents.length > 0 ? "text-yellow-500" :
                      "text-red-500"
                    }`}>
                      {task.state === "PENDING" ? "Processing" : 
                       task.state === "FINISHED" && (!task.result?.failed_documents || task.result.failed_documents.length === 0) ? "Success" : 
                       task.state === "FINISHED" && task.result?.failed_documents && task.result.failed_documents.length > 0 ? "Partially Completed" :
                       "Failed"}
                    </span>
                    <div className="text-xs text-neutral-400 truncate">
                      {task.state === "PENDING" ? "Document upload in progress..." : 
                       task.state === "FINISHED" && (!task.result?.failed_documents || task.result.failed_documents.length === 0) ? 
                        "All documents were successfully processed." : 
                       task.state === "FINISHED" && task.result?.failed_documents && task.result.failed_documents.length > 0 ? 
                        `${task.result.documents.length} document(s) processed successfully, ${task.result.failed_documents.length} failed.` :
                        "Document processing failed."}
                    </div>
                  </div>
                </div>
                <button
                  onClick={() => dismissTask(task.id)}
                  className="text-neutral-500 hover:text-white transition-colors flex-shrink-0 ml-2"
                  title="Dismiss"
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M18 6L6 18M6 6L18 18" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                </button>
              </div>
              
              {/* Timestamp Information */}
              <div className="mt-2 mb-3 text-xs border-b border-neutral-800 pb-3">
                {(() => {
                  const createdAt = task.created_at ? new Date(task.created_at) : null;
                  const formattedTime = createdAt ? 
                    `${createdAt.toLocaleDateString()} at ${createdAt.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}` : 
                    'Just now';
                  
                  const fileCount = task.documents?.length || 0;
                  const successCount = task.result?.total_documents || 
                    (task.state === "FINISHED" ? (task.result?.documents?.length || 0) : 0);
                  const failedCount = task.result?.failed_documents?.length || 0;
                  
                  return (
                    <div className="flex flex-col gap-1 text-neutral-400">
                      <div className="flex items-center gap-2 flex-wrap">
                        <svg className="text-neutral-500 flex-shrink-0" width="12" height="12" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                          <path d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                        </svg>
                        <span className="truncate">{task.state === "PENDING" 
                          ? `Started processing at ${formattedTime}` 
                          : `Completed at ${formattedTime}`}
                        </span>
                      </div>

                      <div className="flex items-center gap-2 flex-wrap">
                        <svg className="text-neutral-500 flex-shrink-0" width="12" height="12" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                          <path d="M9 12h3.75M9 15h3.75M9 18h3.75m3 .75H18a2.25 2.25 0 002.25-2.25V6.108c0-1.135-.845-2.098-1.976-2.192a48.424 48.424 0 00-1.123-.08m-5.801 0c-.065.21-.1.433-.1.664 0 .414.336.75.75.75h4.5a.75.75 0 00.75-.75 2.25 2.25 0 00-.1-.664m-5.8 0A2.251 2.251 0 0113.5 2.25H15c1.012 0 1.867.668 2.15 1.586m-5.8 0c-.376.023-.75.05-1.124.08C9.095 4.01 8.25 4.973 8.25 6.108V8.25m0 0H4.875c-.621 0-1.125.504-1.125 1.125v11.25c0 .621.504 1.125 1.125 1.125h9.75c.621 0 1.125-.504 1.125-1.125V9.375c0-.621-.504-1.125-1.125-1.125H8.25zM6.75 12h.008v.008H6.75V12zm0 3h.008v.008H6.75V15zm0 3h.008v.008H6.75V18z" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                        </svg>
                        <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
                          <span>Total files: <span className="font-medium text-white">{fileCount}</span></span>
                          {task.state !== "PENDING" && (
                            <>
                              <span className="text-neutral-600 hidden sm:inline-block">|</span>
                              <span className="text-green-500">{successCount} successful</span>
                              {failedCount > 0 && (
                                <>
                                  <span className="text-neutral-600 hidden sm:inline-block">|</span>
                                  <span className="text-red-500">{failedCount} failed</span>
                                </>
                              )}
                            </>
                          )}
                        </div>
                      </div>
                    </div>
                  );
                })()}
              </div>
              
              {/* File Information */}
              <div className="text-xs">
                {task.state === "PENDING" && task.documents && task.documents.length > 0 && (
                  <div className="mb-3">
                    <div className="flex items-center justify-between">
                      <p className="text-neutral-400 text-xs mb-1 font-medium">Processing Files</p>
                      {task.documents.length > 5 && (
                        <span className="text-[10px] text-neutral-500">
                          Showing {expandedLists[`pending-${task.id}`] ? task.documents.length : 5} of {task.documents.length}
                        </span>
                      )}
                    </div>
                    <div className={`pl-2 text-neutral-300 text-xs ${task.documents.length > 5 ? 'overflow-y-auto pr-2 custom-scrollbar' : ''} ${expandedLists[`pending-${task.id}`] ? 'max-h-48' : 'max-h-24'}`}>
                      {(expandedLists[`pending-${task.id}`] || task.documents.length <= 5 
                        ? task.documents 
                        : task.documents.slice(0, 5)).map((doc, idx) => (
                        <div key={idx} className="flex items-center gap-1 mb-0.5 truncate">
                          <div className="h-1.5 w-1.5 rounded-full bg-yellow-400 animate-pulse flex-shrink-0"></div>
                          <span className="truncate">{doc}</span>
                        </div>
                      ))}
                      {task.documents.length > 5 && (
                        <button 
                          onClick={() => toggleExpanded(`pending-${task.id}`)}
                          className="text-neutral-400 hover:text-neutral-200 text-[10px] mt-1 cursor-pointer transition-colors flex items-center gap-1"
                        >
                          {expandedLists[`pending-${task.id}`] ? (
                            <>
                              <svg width="10" height="10" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <path d="M19 15L12 9L5 15" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                              </svg>
                              Show fewer files
                            </>
                          ) : (
                            <>
                              <svg width="10" height="10" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <path d="M5 9L12 15L19 9" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                              </svg>
                              + {task.documents.length - 5} more file{task.documents.length - 5 !== 1 ? 's' : ''}
                            </>
                          )}
                        </button>
                      )}
                    </div>
                  </div>
                )}
                
                {/* Show successful documents */}
                {task.state === "FINISHED" && task.result?.documents && task.result.documents.length > 0 && (
                  <div className="mb-3">
                    <div className="flex items-center justify-between">
                      <p className="text-green-500 text-xs mb-1 font-medium flex items-center gap-1">
                        <svg width="12" height="12" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg" className="flex-shrink-0">
                          <path d="M6.00016 10.7799L3.22016 7.99987L2.27349 8.93987L6.00016 12.6665L14.0002 4.66654L13.0602 3.72654L6.00016 10.7799Z" fill="#22C55E"/>
                        </svg>
                        Successful Documents
                      </p>
                      {task.result.documents.length > 5 && (
                        <span className="text-[10px] text-neutral-500">
                          Showing {expandedLists[`success-${task.id}`] ? task.result.documents.length : 5} of {task.result.documents.length}
                        </span>
                      )}
                    </div>
                    <div className={`pl-2 text-neutral-300 text-xs ${task.result.documents.length > 5 ? 'overflow-y-auto pr-2 custom-scrollbar' : ''} ${expandedLists[`success-${task.id}`] ? 'max-h-48' : 'max-h-24'}`}>
                      {(expandedLists[`success-${task.id}`] || task.result.documents.length <= 5 
                        ? task.result.documents 
                        : task.result.documents.slice(0, 5)).map((doc: SuccessfulDocument, idx: number) => (
                        <div key={idx} className="flex items-center gap-1 mb-1 truncate">
                          <span className="text-neutral-500 flex-shrink-0">•</span>
                          <span className="truncate">{doc.document_name}</span>
                          {doc.size_bytes && (
                            <span className="text-neutral-500 text-[10px] ml-1 whitespace-nowrap">
                              ({(doc.size_bytes / 1024).toFixed(1)} KB)
                            </span>
                          )}
                        </div>
                      ))}
                      {task.result.documents.length > 5 && (
                        <button 
                          onClick={() => toggleExpanded(`success-${task.id}`)}
                          className="text-neutral-400 hover:text-neutral-200 text-[10px] mt-1 cursor-pointer transition-colors flex items-center gap-1"
                        >
                          {expandedLists[`success-${task.id}`] ? (
                            <>
                              <svg width="10" height="10" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <path d="M19 15L12 9L5 15" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                              </svg>
                              Show fewer files
                            </>
                          ) : (
                            <>
                              <svg width="10" height="10" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <path d="M5 9L12 15L19 9" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                              </svg>
                              + {task.result.documents.length - 5} more file{task.result.documents.length - 5 !== 1 ? 's' : ''}
                            </>
                          )}
                        </button>
                      )}
                    </div>
                  </div>
                )}
                
                {/* Show failed documents */}
                {task.state !== "PENDING" && task.result?.failed_documents && task.result.failed_documents.length > 0 && (
                  <div className="mb-1">
                    <div className="flex items-center justify-between">
                      <p className="text-red-500 text-xs mb-1 font-medium flex items-center gap-1">
                        <svg width="12" height="12" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg" className="flex-shrink-0">
                          <path d="M8.00016 6.94654L4.97349 3.91987L3.91349 4.97987L6.94016 8.00654L3.91349 11.0332L4.97349 12.0932L8.00016 9.06654L11.0268 12.0932L12.0868 11.0332L9.06016 8.00654L12.0868 4.97987L11.0268 3.91987L8.00016 6.94654Z" fill="#EF4444"/>
                        </svg>
                        Failed Documents ({task.result.failed_documents.length})
                      </p>
                      {task.result.failed_documents.length > 5 && (
                        <span className="text-[10px] text-neutral-500">
                          Showing {expandedLists[`failed-${task.id}`] ? task.result.failed_documents.length : Math.min(task.result.failed_documents.length, 5)} of {task.result.failed_documents.length}
                        </span>
                      )}
                    </div>
                    <div className="pl-2 text-neutral-300 text-xs max-h-40 overflow-y-auto pr-2 custom-scrollbar">
                      {(expandedLists[`failed-${task.id}`] || task.result.failed_documents.length <= 5 
                        ? task.result.failed_documents
                        : task.result.failed_documents.slice(0, 5)).map((doc: FailedDocument, idx: number) => (
                        <div key={idx} className="mb-3 p-2 rounded-md bg-red-500/5 border border-red-500/20">
                          <div className="flex items-center gap-1 mb-1 truncate text-red-400">
                            <svg width="10" height="10" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg" className="flex-shrink-0">
                              <path d="M8.00016 6.94654L4.97349 3.91987L3.91349 4.97987L6.94016 8.00654L3.91349 11.0332L4.97349 12.0932L8.00016 9.06654L11.0268 12.0932L12.0868 11.0332L9.06016 8.00654L12.0868 4.97987L11.0268 3.91987L8.00016 6.94654Z" fill="#EF4444"/>
                            </svg>
                            <span className="font-medium truncate">{doc.document_name}</span>
                          </div>
                          {doc.error_message && (
                            <div className="pl-4 text-[11px] text-red-400/80 mt-1 max-w-full break-words">
                              <span className="font-medium">Error:</span> {doc.error_message}
                            </div>
                          )}
                        </div>
                      ))}
                      {task.result.failed_documents.length > 5 && (
                        <button 
                          onClick={() => toggleExpanded(`failed-${task.id}`)}
                          className="text-neutral-400 hover:text-neutral-200 text-[10px] mt-1 cursor-pointer transition-colors flex items-center gap-1"
                        >
                          {expandedLists[`failed-${task.id}`] ? (
                            <>
                              <svg width="10" height="10" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <path d="M19 15L12 9L5 15" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                              </svg>
                              Show fewer files
                            </>
                          ) : (
                            <>
                              <svg width="10" height="10" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <path d="M5 9L12 15L19 9" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                              </svg>
                              + {task.result.failed_documents.length - 5} more file{task.result.failed_documents.length - 5 !== 1 ? 's' : ''}
                            </>
                          )}
                        </button>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title="Add Source"
      description={modalDescription}
      isLoading={isLoading}
      error={error}
      selectedFiles={selectedFiles}
      submitButtonText="Add Source"
      isSubmitDisabled={selectedFiles.length === 0}
      onFileSelect={handleFileSelect}
      onRemoveFile={removeFile}
      onReset={handleReset}
      onSubmit={handleSubmit}
      metadataSchema={metadataSchema}
      fileMetadata={fileMetadata}
      onMetadataChange={handleMetadataChange}
      fileInputId="sourceFileInput"
      customContent={
        <>
          {collectionSelector}
          <MetadataSchemaEditor
            allowNewField={false}
            schema={metadataSchema ?? []}
            setSchema={setMetadataSchema}
          />
          {renderTaskStatuses()}
        </>
      }
      showFileInput={showDocumentUpload}
    />
  );
}

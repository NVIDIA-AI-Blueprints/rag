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

import React, { useState, useEffect, useRef } from "react";
import AddSourceModal from "./AddSourceModal";
import { useApp } from "@/app/context/AppContext";
import Image from "next/image";
import { APIMetadataField, DocumentMetadata } from "@/types/collections";

interface CollectionItemProps {
  name: string;
  metadataSchema?: APIMetadataField[];
  isSelected: boolean;
  onSelect: () => void;
  onDelete: () => void;
  handleViewFiles: (name: string) => void;
  onDocumentsUpdate: () => void;
  onShowTaskStatus?: () => void;
}

function CollectionItem({
  name,
  metadataSchema,
  isSelected,
  onSelect,
  onDelete,
  handleViewFiles,
  onDocumentsUpdate,
  onShowTaskStatus,
}: CollectionItemProps) {
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const [isAddSourceModalOpen, setIsAddSourceModalOpen] = useState(false);
  const [dropdownPosition, setDropdownPosition] = useState({ top: 0, right: 0 });
  const [showTooltip, setShowTooltip] = useState(false);
  const [tooltipPosition, setTooltipPosition] = useState({ top: 0, left: 0 });
  const indicatorRef = useRef<HTMLDivElement>(null);
  const { pendingTasks } = useApp();

  const [failedDocs, setFailedDocs] = useState<
    { document_name: string; error_message: string }[]
  >([]);

  const hasFailures = failedDocs.length > 0;

  useEffect(() => {
    const key = `failedDocs:${name}`;
    const stored = localStorage.getItem(key);
    if (stored) {
      try {
        setFailedDocs(JSON.parse(stored));
      } catch {
        setFailedDocs([]);
      }
    } else {
      setFailedDocs([]);
    }
  }, [name]);

  const hasPendingTasks = pendingTasks.some(
    task => task.collection_name === name && task.state === "PENDING"
  );

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      const target = event.target as HTMLElement;
      if (isDropdownOpen && !target.closest(".dropdown-container")) {
        setIsDropdownOpen(false);
      }
    };

    if (isDropdownOpen) {
      document.addEventListener("click", handleClickOutside);
    }

    return () => {
      document.removeEventListener("click", handleClickOutside);
    };
  }, [isDropdownOpen]);

  useEffect(() => {
    if (showTooltip && indicatorRef.current) {
      const rect = indicatorRef.current.getBoundingClientRect();
      const idealLeft = rect.right + 10;
      const idealTop = rect.top + rect.height / 2;
      const windowWidth = window.innerWidth;
      const tooltipWidth = 200;

      const leftPos =
        idealLeft + tooltipWidth > windowWidth
          ? rect.left - tooltipWidth - 10
          : idealLeft;

      setTooltipPosition({ top: idealTop, left: leftPos });
    }
  }, [showTooltip]);

  const handleDropdownClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (e.currentTarget) {
      const button = e.currentTarget as HTMLElement;
      const rect = button.getBoundingClientRect();
      setDropdownPosition({
        top: rect.bottom + window.scrollY,
        right: window.innerWidth - rect.right - window.scrollX,
      });
    }
    setIsDropdownOpen(!isDropdownOpen);
  };

  const handleAddSource = (e: React.MouseEvent) => {
    e.stopPropagation();
    setIsDropdownOpen(false);
    setIsAddSourceModalOpen(true);
  };

  const handleDelete = (e: React.MouseEvent) => {
    e.stopPropagation();
    setIsDropdownOpen(false);
    localStorage.removeItem(`failedDocs:${name}`);
    onDelete();
  };

  const handleShowTaskStatus = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (onShowTaskStatus) {
      onShowTaskStatus();
    } else {
      setIsAddSourceModalOpen(true);
    }
  };

  return (
    <div
      className={`group relative flex cursor-pointer flex-col border-b border-neutral-800 px-3 py-2 hover:bg-neutral-900 last:border-b-0 ${
        hasPendingTasks ? "processing-item animate-subtle-pulse" : ""
      }`}
      onClick={onSelect}
    >
      <div className="flex items-center justify-between">
        <div className="flex min-w-0 flex-1 items-center gap-2 overflow-hidden">
          {/* shows selected indicator */}
          <div
            className={`mr-2 flex h-4 w-4 flex-shrink-0 items-center justify-center rounded-[2px] border ${
              isSelected
                ? "border-[var(--nv-green)] bg-[var(--nv-green)]"
                : "border-gray-600"
            }`}
          >
            {isSelected && (
              <svg
                className="w-3 h-3 text-black"
                fill="none"
                stroke="currentColor"
                strokeWidth="3"
                viewBox="0 0 24 24"
              >
                <path d="M5 13l4 4L19 7" />
              </svg>
            )}
          </div>

          <Image
            src="/collection.svg"
            alt="NVIDIA Logo"
            width={20}
            height={20}
            className="flex-shrink-0 invert"
          />
          <div className="flex items-center gap-2 max-w-[180px]">
            <span className="truncate text-sm text-white" title={name}>{name}</span>
            {hasPendingTasks && (
              <div
                ref={indicatorRef}
                className="w-1.5 h-1.5 rounded-full bg-[var(--nv-green)] animate-pulse"
                onMouseEnter={() => setShowTooltip(true)}
                onMouseLeave={() => setShowTooltip(false)}
                onClick={handleShowTaskStatus}
              />
            )}
          </div>
        </div>
        {hasFailures && (
          <div
            ref={indicatorRef}
            className="text-red-400 cursor-help"
          >⚠</div>
        )}
        <button
          onClick={handleDropdownClick}
          className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full text-gray-400 hover:bg-neutral-800 hover:text-white"
        >
          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
            <circle cx="12" cy="12" r="2" />
            <circle cx="12" cy="5" r="2" />
            <circle cx="12" cy="19" r="2" />
          </svg>
        </button>
      </div>

      {(hasPendingTasks || hasFailures) && showTooltip && (
        <div
          className="fixed z-[9999] bg-neutral-800/95 text-xs text-white px-3 py-2 rounded-md shadow-lg backdrop-blur-sm border border-red-500 max-w-xs whitespace-pre-wrap pointer-events-none transform -translate-y-1/2"
          style={{ top: `${tooltipPosition.top}px`, left: `${tooltipPosition.left}px` }}
        >
          {hasPendingTasks && (
            <>
              <div className="font-medium">Processing files</div>
              <div className="text-[10px] text-neutral-400 mt-0.5">Click to view status</div>
            </>
          )}
          {hasFailures && (
            <>
              <div className="font-semibold mb-1">Failed documents</div>
              <ul className="list-disc list-inside space-y-0.5 text-[11px]">
                {failedDocs.map((doc, idx) => (
                  <li key={idx}><strong>{doc.document_name}:</strong> {doc.error_message}</li>
                ))}
              </ul>
            </>
          )}
        </div>
      )}

      {hasFailures && (
        <div
          ref={indicatorRef}
          className="text-red-400 cursor-help"
          onMouseEnter={() => setShowTooltip(true)}
          onMouseLeave={() => setShowTooltip(false)}
        >
          ⚠
        </div>
      )}

      {isDropdownOpen && (
        <div className="dropdown-container fixed z-50 w-48" style={{ top: `${dropdownPosition.top}px`, right: `${dropdownPosition.right}px` }}>
          <div className="rounded-md border border-neutral-800 bg-neutral-900 shadow-lg">
            <button onClick={handleAddSource} className="flex w-full items-center px-4 py-2 text-sm text-white hover:bg-neutral-800">Add Source</button>
            <button onClick={(e) => { e.stopPropagation(); setIsDropdownOpen(false); handleViewFiles(name); }} className="flex w-full items-center px-4 py-2 text-sm text-white hover:bg-neutral-800">View Files</button>
            {hasPendingTasks && (
              <button onClick={handleShowTaskStatus} className="flex w-full items-center px-4 py-2 text-sm font-medium text-[var(--nv-green)] hover:bg-neutral-800">
                <svg className="mr-2 h-3.5 w-3.5 text-[var(--nv-green)]" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M12 4V2a10 10 0 0 0-10 10h2a8 8 0 0 1 8-8Z">
                    <animateTransform attributeName="transform" attributeType="XML" type="rotate" from="0 12 12" to="360 12 12" dur="1s" repeatCount="indefinite" />
                  </path>
                </svg>
                View Processing Status
              </button>
            )}
            <button onClick={handleDelete} className="flex w-full items-center px-4 py-2 text-sm text-red-500 hover:bg-neutral-800">Delete Collection</button>
          </div>
        </div>
      )}

      <AddSourceModal isOpen={isAddSourceModalOpen} onClose={() => setIsAddSourceModalOpen(false)} collectionName={name} onDocumentsUpdate={onDocumentsUpdate} />
    </div>
  );
}

function arePropsEqual(prevProps: CollectionItemProps, nextProps: CollectionItemProps) {
  return prevProps.name === nextProps.name && prevProps.isSelected === nextProps.isSelected;
}

export default React.memo(CollectionItem, arePropsEqual);

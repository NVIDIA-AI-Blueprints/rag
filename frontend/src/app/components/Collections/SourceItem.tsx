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
import React from "react";
import Image from "next/image";
import { DocumentMetadata, UIMetadataField } from "@/types/collections";

interface SourceItemProps {
  name: string;
  metadata?: UIMetadataField;
  metadataSchema?: UIMetadataField[];
  onDelete: () => void;
}

function SourceItem({ name, metadata, metadataSchema = [], onDelete }: SourceItemProps) {
  const handleDelete = (e: React.MouseEvent) => {
    e.stopPropagation();
    onDelete();
  };

  const shouldShowField = (key: string, value: string | null | undefined) => {
    if (value == null) return false;
    const field = metadataSchema.find((f) => f.name === key);
    const isEmpty = value.trim() === "" || value.toLowerCase() === "nan";
    if (!field) return !isEmpty; // unknown field, show if not empty
    if (field.optional && isEmpty) return false;
    return true;
  };

  return (
    <div className="flex flex-col border-b border-neutral-800 hover:bg-neutral-900 cursor-pointer">
      <div className="flex items-center justify-between px-3 py-2">
        <div className="flex min-w-0 flex-1 items-center gap-2 overflow-hidden">
          <Image
            src="/document.svg"
            alt="Document"
            width={16}
            height={16}
            className="flex-shrink-0"
          />
          <span
            className="max-w-[180px] truncate text-sm text-white"
            title={name}
          >
            {name}
          </span>
        </div>
        <button
          onClick={handleDelete}
          className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full text-gray-400 hover:text-red-500 hover:bg-neutral-800"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M3 6h18" />
            <path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6" />
            <path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2" />
          </svg>
        </button>
      </div>

      {metadata && (
        <div className="text-xs text-neutral-400 px-6 pb-2 space-y-1">
          {Object.entries(metadata)
            .filter(([key, value]) => shouldShowField(key, value))
            .map(([key, value], idx) => (
              <div key={idx} className="flex gap-2">
                <span className="text-white">{key}:</span>
                <span className="truncate">
                  {typeof value === "string" &&
                  value.match(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$/)
                    ? new Date(value).toLocaleString(undefined, {
                        year: "numeric",
                        month: "short",
                        day: "numeric",
                        hour: "numeric",
                        minute: "2-digit",
                        hour12: true,
                      })
                    : value}
                </span>
              </div>
            ))}
        </div>
      )}
    </div>
  );
}

export default React.memo(SourceItem);

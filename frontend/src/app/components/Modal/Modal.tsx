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

import { useState } from "react";
import Image from "next/image";
import { ReactNode } from "react";
import { UIMetadataField } from "@/types/collections";

interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  title: string;
  description?: string;
  isLoading?: boolean;
  error?: string | null;
  selectedFiles?: File[];
  fileMetadata?: Record<string, Record<string, string>>;
  submitButtonText?: string;
  isSubmitDisabled?: boolean;
  onFileSelect?: (files: File[]) => void;
  onRemoveFile?: (index: number) => void;
  onMetadataChange?: (filename: string, field: string, value: string) => void;
  onReset?: () => void;
  onSubmit?: () => void;
  metadataSchema?: UIMetadataField[];
  fileInputId?: string;
  customContent?: ReactNode;
  showFileInput?: boolean;
  hideActionButtons?: boolean;
}

export default function Modal({
  isOpen,
  onClose,
  title,
  description,
  isLoading = false,
  error = null,
  selectedFiles = [],
  fileMetadata = {},
  submitButtonText = "Submit",
  isSubmitDisabled = false,
  onFileSelect,
  onRemoveFile,
  onMetadataChange,
  onReset,
  onSubmit,
  metadataSchema,
  fileInputId = "fileInput",
  customContent,
  showFileInput = true,
  hideActionButtons = false,
}: ModalProps) {
  const schema = metadataSchema || [];

  if (!isOpen) return null;

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && onFileSelect) {
      const files = Array.from(e.target.files);
      onFileSelect(files);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm overflow-y-auto py-6"
      onClick={onClose}
    >
      <div
        className="mx-4 w-full max-w-lg rounded-lg bg-[#1A1A1A] flex flex-col max-h-[90vh]"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-neutral-800 p-4 shrink-0">
          <h2 className="text-lg font-semibold text-white">{title}</h2>
          <button
            onClick={onClose}
            className="text-gray-400 transition-colors hover:text-white"
            aria-label="Close modal"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="24"
              height="24"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>
        <div className="p-4 text-white overflow-y-auto flex-grow custom-scrollbar">
          {description && (
            <div className="mb-4">
              <p className="text-sm text-gray-400">{description}</p>
            </div>
          )}

          {customContent}

          {showFileInput && (
            <div className="mb-4">
              <label className="mb-2 block text-sm font-medium">
                Source Files
              </label>
              <div
                className="cursor-pointer rounded-lg border-2 border-dashed border-gray-600 p-4 text-center transition-colors hover:border-[var(--nv-green)]"
                onClick={() => document.getElementById(fileInputId)?.click()}
              >
                <input
                  id={fileInputId}
                  type="file"
                  multiple
                  onChange={handleFileSelect}
                  className="hidden"
                  accept=".bmp,.docx,.html,.jpeg,.json,.md,.pdf,.png,.pptx,.sh,.tiff,.txt,.mp3,.wav"
                />
                <div className="flex flex-col items-center justify-center">
                  <Image src="/file.svg" alt="Upload files" width={32} height={32} className="mb-2 opacity-50" />
                  <p className="mb-1 text-sm text-gray-400">Click to upload or drag and drop</p>
                  <p className="text-xs text-gray-500">
                    Supported file types: BMP, DOCX, HTML, JPEG, JSON, MD, PDF, PNG, PPTX, SH, TIFF, TXT, MP3, WAV
                  </p>
                </div>
              </div>
            </div>
          )}

          {selectedFiles.length > 0 && (
            <div className="mb-4">
              <h3 className="mb-2 text-sm font-medium">Selected Files</h3>
              <div className="scrollbar-thin scrollbar-track-neutral-900 scrollbar-thumb-neutral-700 max-h-96 space-y-4 overflow-y-auto pr-1">
                {selectedFiles.map((file, index) => (
                  <div key={index} className="rounded-md bg-neutral-800 p-3">
                    <div className="flex items-center justify-between">
                      <span className="truncate text-sm font-medium text-white">{file.name}</span>
                      <button
                        onClick={() => onRemoveFile && onRemoveFile(index)}
                        className="text-gray-400 transition-colors hover:text-white"
                        aria-label={`Remove ${file.name}`}
                      >
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <line x1="18" y1="6" x2="6" y2="18" />
                          <line x1="6" y1="6" x2="18" y2="18" />
                        </svg>
                      </button>
                    </div>

                    {schema?.length > 0 && (
                      <div className="mt-3 space-y-3">
                        {schema.map((field) => {
                          const key = `${file.name}-${field.name}`;
                          const value = fileMetadata?.[file.name]?.[field.name] || "";

                          const inputProps = {
                            className: "w-full rounded-md px-3 py-1 text-sm text-white bg-neutral-700",
                            value,
                            onChange: (e: React.ChangeEvent<HTMLInputElement>) =>
                              onMetadataChange?.(file.name, field.name, e.target.value),
                          };

                          return (
                            <div key={key}>
                              <label className="block text-xs text-neutral-400 mb-1">
                                {field.name}{" "}
                                <span className="text-neutral-500 text-[10px]">({field.type})</span>
                              </label>
                              {field.type === "datetime" ? (
                                <input
                                  type="datetime-local"
                                  className={inputProps.className}
                                  value={value ? value.slice(0, 16) : ""}
                                  onChange={(e) => {
                                    const val = e.target.value; // "2025-06-12T14:30"
                                    const normalized = val ? `${val}:00` : ""; // append seconds
                                    onMetadataChange?.(file.name, field.name, normalized);
                                  }}
                                />
                              ) : (
                                <input type="text" {...inputProps} />
                              )}
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {error && (
            <div className="mb-4 rounded-md bg-red-900/50 p-3 text-sm text-red-200">{error}</div>
          )}
        </div>

        {!hideActionButtons && (
          <div className="mt-2 flex items-center justify-end space-x-3 p-4 border-t border-neutral-800 shrink-0">
            <button
              type="button"
              className="px-4 py-2 text-sm font-medium text-white bg-transparent border border-neutral-600 rounded-full hover:bg-neutral-800 focus:outline-none disabled:opacity-50 disabled:cursor-not-allowed"
              onClick={onReset}
              disabled={isLoading}
            >
              Reset
            </button>
            <button
              type="button"
              className="px-4 py-2 text-sm font-medium text-white bg-[var(--nv-green)] rounded-full hover:bg-opacity-90 focus:outline-none disabled:opacity-50 disabled:cursor-not-allowed"
              onClick={onSubmit}
              disabled={isSubmitDisabled || isLoading}
            >
              {isLoading ? (
                <span className="flex items-center">
                  <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Processing...
                </span>
              ) : (
                submitButtonText
              )}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

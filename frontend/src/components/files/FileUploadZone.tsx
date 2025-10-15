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

import { useRef, useCallback } from "react";
import { useDragAndDrop } from "../../hooks/useDragAndDrop";

interface FileUploadZoneProps {
  acceptedTypes: string[];
  maxFileSize: number;
  audioFileMaxSize?: number;
  onFilesSelected: (files: FileList) => void;
}

const UploadIcon = () => (
  <svg className="w-12 h-12 text-neutral-500 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
  </svg>
);

export const FileUploadZone = ({ 
  acceptedTypes, 
  maxFileSize, 
  audioFileMaxSize,
  onFilesSelected 
}: FileUploadZoneProps) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  const { isDragOver, dragHandlers } = useDragAndDrop({
    onFilesDropped: onFilesSelected,
  });

  const handleChooseFiles = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      onFilesSelected(files);
    }
    // Reset input value to allow selecting the same file again
    e.target.value = '';
  }, [onFilesSelected]);

  return (
    <div
      className={`relative border-2 border-dashed rounded-lg p-8 text-center transition-colors cursor-pointer ${
        isDragOver
          ? 'border-[var(--nv-green)] bg-[var(--nv-green)]/5'
          : 'border-neutral-600 hover:border-neutral-500'
      }`}
      {...dragHandlers}
      onClick={handleChooseFiles}
    >
      <div className="space-y-2">
        <UploadIcon />
        <p className="text-white">
          <button
            onClick={(e) => {
              e.stopPropagation(); // Prevent double trigger
              handleChooseFiles();
            }}
            className="text-white underline hover:text-[var(--nv-green)] transition-colors font-medium"
          >
            Choose files
          </button>
          {' '}or drag and drop them here.
        </p>
        <p className="text-sm text-gray-400">
          Accepted: {acceptedTypes.join(', ')} • Up to {maxFileSize} MB • Max 100 files per batch
          {audioFileMaxSize && audioFileMaxSize !== maxFileSize && (
            <span> • Audio files (.mp3, .wav): up to {audioFileMaxSize} MB</span>
          )}
        </p>
      </div>

      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept={acceptedTypes.join(',')}
        onChange={handleFileSelect}
        className="hidden"
      />
    </div>
  );
}; 
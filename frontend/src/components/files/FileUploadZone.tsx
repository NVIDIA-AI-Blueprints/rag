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
import { Text, Flex, Stack } from "@kui/react";
import { Upload } from "lucide-react";

interface FileUploadZoneProps {
  acceptedTypes: string[];
  maxFileSize: number;
  audioFileMaxSize?: number;
  onFilesSelected: (files: FileList) => void;
}

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
      style={{
        border: `2px dashed ${isDragOver ? 'var(--color-brand)' : 'var(--border-color-base)'}`,
        borderRadius: 'var(--border-radius-lg)',
        padding: 'var(--spacing-density-xl)',
        textAlign: 'center',
        cursor: 'pointer',
        backgroundColor: isDragOver ? 'var(--color-brand-alpha-5)' : 'transparent',
        transition: 'all 0.2s'
      }}
      {...dragHandlers}
      onClick={handleChooseFiles}
    >
      <Stack gap="density-sm" align="center">
        <Upload size={48} style={{ color: 'var(--text-color-subtle)', marginBottom: 'var(--spacing-density-sm)' }} />
        <Flex gap="density-xs" align="center">
          <button
            onClick={(e) => {
              e.stopPropagation();
              handleChooseFiles();
            }}
            style={{
              background: 'none',
              border: 'none',
              textDecoration: 'underline',
              cursor: 'pointer',
              color: 'var(--text-color-default)',
              fontWeight: 500
            }}
          >
            Choose files
          </button>
          <Text kind="body/regular/md">or drag and drop them here.</Text>
        </Flex>
        <Text kind="body/regular/sm" style={{ color: 'var(--text-color-subtle)' }}>
          Accepted: {acceptedTypes.join(', ')} • Up to {maxFileSize} MB • Max 100 files per batch
          {audioFileMaxSize && audioFileMaxSize !== maxFileSize && (
            <span> • Audio files (.mp3, .wav): up to {audioFileMaxSize} MB</span>
          )}
        </Text>
      </Stack>

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

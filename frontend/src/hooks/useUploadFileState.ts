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

import { useState, useCallback } from "react";
import { useFileValidation } from "./useFileValidation";
import { useFileUtils } from "./useFileUtils";

/**
 * Interface representing a file in the upload state.
 * Contains the file, validation status, and upload progress information.
 */
export interface UploadFile {
  id: string;
  file: File;
  status: 'uploading' | 'uploaded' | 'error';
  progress: number;
  errorMessage?: string;
}

/**
 * Options for configuring the upload file state hook.
 */
interface UploadFileStateOptions {
  acceptedTypes?: string[];
  maxFileSize?: number; // in MB
  maxFiles?: number;
  audioFileMaxSize?: number; // in MB, overrides maxFileSize for audio files
  onFilesChange?: (files: File[]) => void;
}

/**
 * Custom hook for managing file upload state with validation and progress tracking.
 * Provides methods to add, remove, and clear files with built-in validation.
 * 
 * @param options Configuration options for file validation and limits
 * @returns Object containing upload files and methods to manipulate them
 * 
 * @example
 * ```tsx
 * const { uploadFiles, addFiles, removeFile } = useUploadFileState({
 *   acceptedTypes: ['.pdf', '.doc'],
 *   maxFileSize: 10,
 *   maxFiles: 5,
 *   onFilesChange: (files) => console.log('Files changed:', files)
 * });
 * ```
 */
export const useUploadFileState = ({
  acceptedTypes,
  maxFileSize,
  maxFiles = 100,
  audioFileMaxSize,
  onFilesChange
}: UploadFileStateOptions) => {
  const [uploadFiles, setUploadFiles] = useState<UploadFile[]>([]);
  const { validateFile } = useFileValidation({ 
    acceptedTypes: acceptedTypes || [], 
    maxFileSize: maxFileSize || 10, 
    audioFileMaxSize 
  });
  const { generateFileId } = useFileUtils();

  const addFiles = useCallback((files: FileList | File[]) => {
    const newFilesToAdd = Array.from(files);
    
    setUploadFiles(prev => {
      // Get existing file names to avoid duplicates
      const existingFileNames = new Set(prev.map(f => f.file.name));
      
      // Filter out duplicates and respect max limit
      const uniqueNewFiles = newFilesToAdd.filter(file => !existingFileNames.has(file.name));
      const remainingSlots = maxFiles - prev.length;
      const filesToProcess = uniqueNewFiles.slice(0, remainingSlots);
      
      // Create upload file objects
      const newUploadFiles = filesToProcess.map(file => {
        const error = validateFile(file);
        return {
          id: generateFileId(),
          file,
          status: error ? 'error' : 'uploaded',
          progress: error ? 0 : 100,
          errorMessage: error || undefined,
        } as UploadFile;
      });
      
      const updatedFiles = [...prev, ...newUploadFiles];
      
      // Call parent with all valid files
      const allValidFiles = updatedFiles
        .filter(f => f.status !== 'error')
        .map(f => f.file);
      console.log('🔵 useUploadFileState: calling onFilesChange with', allValidFiles.length, 'files');
      onFilesChange?.(allValidFiles);
      
      return updatedFiles;
    });
  }, [maxFiles, validateFile, generateFileId, onFilesChange]);

  const replaceFiles = useCallback((files: FileList | File[]) => {
    const newFiles: UploadFile[] = [];
    const validFiles: File[] = [];
    
    Array.from(files).slice(0, maxFiles).forEach((file) => {
      const error = validateFile(file);
      const uploadFile: UploadFile = {
        id: generateFileId(),
        file,
        status: error ? 'error' : 'uploaded',
        progress: error ? 0 : 100,
        errorMessage: error || undefined,
      };
      newFiles.push(uploadFile);
      if (!error) {
        validFiles.push(file);
      }
    });

    setUploadFiles(newFiles);
    onFilesChange?.(validFiles);
  }, [maxFiles, validateFile, onFilesChange, generateFileId]);

  const removeFile = useCallback((id: string) => {
    setUploadFiles(prev => {
      const updatedFiles = prev.filter(f => f.id !== id);
      
      // Calculate remaining valid files
      const remainingValidFiles = updatedFiles
        .filter(f => f.status !== 'error')
        .map(f => f.file);
      
      // Call onFilesChange with the correct remaining files
      onFilesChange?.(remainingValidFiles);
      
      return updatedFiles;
    });
  }, [onFilesChange]);

  const clearAllFiles = useCallback(() => {
    setUploadFiles([]);
    onFilesChange?.([]);
  }, [onFilesChange]);

  const getValidFiles = useCallback(() => {
    return uploadFiles
      .filter(uploadFile => uploadFile.status !== 'error')
      .map(uploadFile => uploadFile.file);
  }, [uploadFiles]);

  return {
    uploadFiles,
    addFiles,
    replaceFiles,
    removeFile,
    clearAllFiles,
    getValidFiles,
  };
};
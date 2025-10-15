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

import { useCallback } from "react";

/**
 * Configuration options for file validation.
 */
interface FileValidationOptions {
  acceptedTypes: string[];
  maxFileSize: number; // in MB
  audioFileMaxSize?: number; // in MB, defaults to maxFileSize if not provided
}

/**
 * Custom hook for validating uploaded files against type and size constraints.
 * 
 * @param options - File validation configuration
 * @returns Object with validation functions for individual files and file lists
 * 
 * @example
 * ```tsx
 * const { validateFile, validateFiles } = useFileValidation({
 *   acceptedTypes: ['.pdf', '.docx'],
 *   maxFileSize: 10 // 10MB
 * });
 * const error = validateFile(file);
 * ```
 */
export const useFileValidation = ({ acceptedTypes, maxFileSize, audioFileMaxSize }: FileValidationOptions) => {
  
  const validateFile = useCallback((file: File): string | null => {
    const fileExtension = '.' + file.name.split('.').pop()?.toLowerCase();
    
    if (!acceptedTypes.includes(fileExtension)) {
      return `File type not supported. Accepted: ${acceptedTypes.join(', ')}`;
    }
    
    // Check if this is an audio file
    const isAudioFile = ['.mp3', '.wav'].includes(fileExtension);
    const sizeLimit = isAudioFile ? (audioFileMaxSize || maxFileSize) : maxFileSize;
    
    if (file.size > sizeLimit * 1024 * 1024) {
      if (isAudioFile && audioFileMaxSize) {
        return `Audio file too large. Max size for audio files: ${audioFileMaxSize}MB`;
      }
      return `File too large. Max size: ${maxFileSize}MB`;
    }
    
    return null;
  }, [acceptedTypes, maxFileSize, audioFileMaxSize]);

  const validateFiles = useCallback((files: File[]): { valid: File[]; invalid: Array<{ file: File; error: string }> } => {
    const valid: File[] = [];
    const invalid: Array<{ file: File; error: string }> = [];

    files.forEach(file => {
      const error = validateFile(file);
      if (error) {
        invalid.push({ file, error });
      } else {
        valid.push(file);
      }
    });

    return { valid, invalid };
  }, [validateFile]);

  return {
    validateFile,
    validateFiles,
  };
}; 
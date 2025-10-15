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

/**
 * Configuration options for the drag and drop hook.
 */
interface DragAndDropOptions {
  onFilesDropped: (files: FileList) => void;
}

/**
 * Custom hook for handling drag and drop file operations.
 * 
 * @param options - Drag and drop configuration with file drop handler
 * @returns Object with drag state and event handlers for drag and drop functionality
 * 
 * @example
 * ```tsx
 * const { isDragOver, handleDragOver, handleDragLeave, handleDrop } = useDragAndDrop({
 *   onFilesDropped: (files) => console.log('Files dropped:', files)
 * });
 * ```
 */
export const useDragAndDrop = ({ onFilesDropped }: DragAndDropOptions) => {
  const [isDragOver, setIsDragOver] = useState(false);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      onFilesDropped(files);
    }
  }, [onFilesDropped]);

  return {
    isDragOver,
    dragHandlers: {
      onDragOver: handleDragOver,
      onDragLeave: handleDragLeave,
      onDrop: handleDrop,
    },
  };
}; 
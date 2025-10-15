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

import type { ReactElement } from 'react';

/**
 * Props interface for file icon components.
 */
export interface FileIconProps {
  className?: string;
  size?: 'sm' | 'md' | 'lg';
  monochrome?: boolean;
}

/**
 * Custom hook for generating file type icons based on file extensions.
 * 
 * Provides a comprehensive set of file icons for different file types
 * including documents, images, code files, and archives. Returns appropriate
 * SVG icons with customizable size and styling options.
 * 
 * @returns Object with icon getter function for file extensions
 * 
 * @example
 * ```tsx
 * const { getFileIcon } = useFileIcons();
 * const pdfIcon = getFileIcon('.pdf', { size: 'lg' });
 * ```
 */
export function useFileIcons() {
  const getSizeClasses = (size: FileIconProps['size'] = 'md') => {
    switch (size) {
      case 'sm': return 'w-4 h-4';
      case 'md': return 'w-5 h-5';
      case 'lg': return 'w-8 h-8';
      default: return 'w-5 h-5';
    }
  };

  const getFileIconByExtension = (fileName: string, props: FileIconProps = {}): ReactElement => {
    const { className = '', size = 'md', monochrome = false } = props;
    const sizeClasses = getSizeClasses(size);
    const extension = fileName.split('.').pop()?.toLowerCase();
    
    // Use gray color when monochrome is enabled
    const getIconColor = (defaultColor: string) => monochrome ? 'text-gray-400' : defaultColor;

    switch (extension) {
      case 'pdf':
        return (
          <svg className={`${sizeClasses} ${getIconColor('text-red-400')} ${className}`} fill="currentColor" viewBox="0 0 24 24">
            <path d="M14,2H6A2,2 0 0,0 4,4V20A2,2 0 0,0 6,22H18A2,2 0 0,0 20,20V8L14,2M18,20H6V4H13V9H18V20M8,11H16V13H8V11M8,15H13V17H8V15Z" />
          </svg>
        );

      case 'docx':
        return (
          <svg className={`${sizeClasses} ${getIconColor('text-blue-500')} ${className}`} fill="currentColor" viewBox="0 0 24 24">
            <path d="M14,2H6A2,2 0 0,0 4,4V20A2,2 0 0,0 6,22H18A2,2 0 0,0 20,20V8L14,2M18,20H6V4H13V9H18V20M8,11H16V13H8V11M8,15H13V17H8V15Z" />
          </svg>
        );

      case 'pptx':
        return (
          <svg className={`${sizeClasses} ${getIconColor('text-orange-500')} ${className}`} fill="currentColor" viewBox="0 0 24 24">
            <path d="M14,2H6A2,2 0 0,0 4,4V20A2,2 0 0,0 6,22H18A2,2 0 0,0 20,20V8L14,2M18,20H6V4H13V9H18V20M8,11H16V13H8V11M8,15H13V17H8V15Z" />
          </svg>
        );

      case 'jpg':
      case 'jpeg':
      case 'png':
      case 'bmp':
      case 'tiff':
        return (
          <svg className={`${sizeClasses} ${getIconColor('text-purple-400')} ${className}`} fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
          </svg>
        );

      case 'mp3':
      case 'wav':
        return (
          <svg className={`${sizeClasses} ${getIconColor('text-pink-400')} ${className}`} fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
          </svg>
        );

      case 'txt':
      case 'md':
        return (
          <svg className={`${sizeClasses} ${getIconColor('text-gray-400')} ${className}`} fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
        );

      case 'json':
        return (
          <svg className={`${sizeClasses} ${getIconColor('text-teal-400')} ${className}`} fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
          </svg>
        );

      case 'html':
        return (
          <svg className={`${sizeClasses} ${getIconColor('text-cyan-400')} ${className}`} fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
          </svg>
        );

      case 'sh':
        return (
          <svg className={`${sizeClasses} ${getIconColor('text-gray-300')} ${className}`} fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M8 9l3 3-3 3m5 0h3M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
          </svg>
        );

      default:
        return (
          <svg className={`${sizeClasses} ${getIconColor('text-gray-400')} ${className}`} fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
        );
    }
  };

  const getDocumentTypeIcon = (documentType: string, props: FileIconProps = {}): ReactElement => {
    const { className = '', size = 'md', monochrome = false } = props;
    const sizeClasses = getSizeClasses(size);
    
    // Use gray color when monochrome is enabled
    const getIconColor = (defaultColor: string) => monochrome ? 'text-gray-400' : defaultColor;

    switch (documentType.toLowerCase()) {
      case 'image':
        return (
          <svg className={`${sizeClasses} ${getIconColor('text-purple-400')} ${className}`} fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
          </svg>
        );

      case 'chart':
        return (
          <svg className={`${sizeClasses} ${getIconColor('text-green-400')} ${className}`} fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
        );

      case 'table':
        return (
          <svg className={`${sizeClasses} ${getIconColor('text-blue-400')} ${className}`} fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M3 10h18M3 14h18m-9-4v8m-7 0V4a1 1 0 011-1h16a1 1 0 011 1v16a1 1 0 01-1 1H5a1 1 0 01-1-1V10z" />
          </svg>
        );

      case 'video':
        return (
          <svg className={`${sizeClasses} ${getIconColor('text-indigo-400')} ${className}`} fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
          </svg>
        );

      case 'audio':
        return (
          <svg className={`${sizeClasses} ${getIconColor('text-pink-400')} ${className}`} fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
          </svg>
        );

      case 'text':
      case 'document':
      default:
        return (
          <svg className={`${sizeClasses} ${getIconColor('text-gray-400')} ${className}`} fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
        );
    }
  };

  return {
    getFileIconByExtension,
    getDocumentTypeIcon,
  };
} 
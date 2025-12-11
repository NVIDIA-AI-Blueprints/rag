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
import { 
  FileText, 
  Image, 
  Music, 
  Code, 
  File, 
  BarChart3, 
  Table, 
  Video 
} from 'lucide-react';

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
 */
export function useFileIcons() {
  const getSize = (size: FileIconProps['size'] = 'md') => {
    switch (size) {
      case 'sm': return 16;
      case 'md': return 20;
      case 'lg': return 32;
      default: return 20;
    }
  };

  const getFileIconByExtension = (fileName: string, props: FileIconProps = {}): ReactElement => {
    const { size = 'md', monochrome = false } = props;
    const iconSize = getSize(size);
    const extension = fileName.split('.').pop()?.toLowerCase();
    
    const getColor = (defaultColor: string) => 
      monochrome ? 'var(--text-color-subtle)' : defaultColor;

    switch (extension) {
      case 'pdf':
        return <FileText size={iconSize} style={{ color: getColor('var(--feedback-color-danger)') }} />;
      case 'docx':
        return <FileText size={iconSize} style={{ color: getColor('#3b82f6') }} />;
      case 'pptx':
        return <FileText size={iconSize} style={{ color: getColor('#f97316') }} />;
      case 'jpg':
      case 'jpeg':
      case 'png':
      case 'bmp':
      case 'tiff':
        return <Image size={iconSize} style={{ color: getColor('#a855f7') }} />;
      case 'mp3':
      case 'wav':
        return <Music size={iconSize} style={{ color: getColor('#ec4899') }} />;
      case 'txt':
      case 'md':
        return <FileText size={iconSize} style={{ color: getColor('var(--text-color-subtle)') }} />;
      case 'json':
      case 'html':
      case 'sh':
        return <Code size={iconSize} style={{ color: getColor('#14b8a6') }} />;
      default:
        return <File size={iconSize} style={{ color: getColor('var(--text-color-subtle)') }} />;
    }
  };

  const getDocumentTypeIcon = (documentType: string, props: FileIconProps = {}): ReactElement => {
    const { size = 'md', monochrome = false } = props;
    const iconSize = getSize(size);
    
    const getColor = (defaultColor: string) => 
      monochrome ? 'var(--text-color-subtle)' : defaultColor;

    switch (documentType.toLowerCase()) {
      case 'image':
        return <Image size={iconSize} style={{ color: getColor('#a855f7') }} />;
      case 'chart':
        return <BarChart3 size={iconSize} style={{ color: getColor('#22c55e') }} />;
      case 'table':
        return <Table size={iconSize} style={{ color: getColor('#3b82f6') }} />;
      case 'video':
        return <Video size={iconSize} style={{ color: getColor('#6366f1') }} />;
      case 'audio':
        return <Music size={iconSize} style={{ color: getColor('#ec4899') }} />;
      case 'text':
      case 'document':
      default:
        return <FileText size={iconSize} style={{ color: getColor('var(--text-color-subtle)') }} />;
    }
  };

  return {
    getFileIconByExtension,
    getDocumentTypeIcon,
  };
}

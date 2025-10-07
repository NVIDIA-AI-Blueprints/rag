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

import { useCallback, useState } from "react";
import { useFileIcons } from "../../hooks/useFileIcons";
import { useFileUtils } from "../../hooks/useFileUtils";
import { FileMetadataForm } from "./FileMetadataForm";
import { ConfirmationModal } from "../modals/ConfirmationModal";
import type { UploadFile } from "../../hooks/useUploadFileState";
import { Block, Flex, Text, Button } from "@kui/react";

interface FileItemProps {
  uploadFile: UploadFile;
  onRemove: (id: string) => void;
}

const RemoveIcon = () => (
  <svg style={{ width: '16px', height: '16px', color: 'white' }} fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
  </svg>
);

const ErrorIcon = () => (
  <Flex 
    align="center" 
    justify="center"
    style={{ 
      width: '32px', 
      height: '32px', 
      border: '1px solid var(--border-color-base)', 
      borderRadius: 'var(--radius-md)'
    }}
  >
    <svg style={{ width: '16px', height: '16px', color: 'var(--text-color-subtle)' }} fill="currentColor" viewBox="0 0 20 20">
      <path d="M4 18h12V6l-4-4H4v16zm6-10h2v6h-2V8z"/>
    </svg>
  </Flex>
);

const WarningIcon = () => (
  <svg style={{ width: '12px', height: '12px', color: 'var(--text-color-feedback-danger)' }} fill="currentColor" viewBox="0 0 20 20">
    <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
  </svg>
);

const FileIcon = ({ uploadFile }: { uploadFile: UploadFile }) => {
  const { getFileIconByExtension } = useFileIcons();
  
  if (uploadFile.status === 'error') {
    return <ErrorIcon />;
  }
  
  return getFileIconByExtension(uploadFile.file.name, { size: 'lg' });
};

const FileStatus = ({ uploadFile }: { uploadFile: UploadFile }) => {
  if (uploadFile.status === 'uploaded') {
    return (
      <>
        <Text kind="body/regular/sm" style={{ color: '#cccccc' }}>•</Text>
        <Text kind="body/regular/sm" style={{ color: 'var(--text-color-feedback-success)' }}>Ready</Text>
      </>
    );
  }
  
  if (uploadFile.status === 'error' && uploadFile.errorMessage) {
    return (
      <>
        <Text kind="body/regular/sm" style={{ color: '#cccccc' }}>•</Text>
        <Flex 
          align="center" 
          gap="density-xs"
          style={{ 
            padding: '4px 8px', 
            backgroundColor: 'var(--background-color-feedback-danger)', 
            borderRadius: 'var(--radius-md)', 
            border: '1px solid var(--border-color-feedback-danger)' 
          }}
        >
          <WarningIcon />
          <Text kind="body/bold/sm" style={{ color: 'var(--text-color-feedback-danger-inverse)' }}>
            {uploadFile.errorMessage}
          </Text>
        </Flex>
      </>
    );
  }
  
  return null;
};

const FileHeader = ({ uploadFile, onRemove }: FileItemProps) => {
  const [showRemoveModal, setShowRemoveModal] = useState(false);
  
  const handleRemoveClick = useCallback(() => {
    setShowRemoveModal(true);
  }, []);
  
  const handleConfirmRemove = useCallback(() => {
    onRemove(uploadFile.id);
  }, [uploadFile.id, onRemove]);

  return (
    <Flex justify="between" align="center" gap="density-sm">
      <Text 
        kind="body/bold/md" 
        style={{ 
          color: '#ffffff',
          wordBreak: 'break-word',
          lineHeight: '1.4'
        }}
      >
        {uploadFile.file.name}
      </Text>
      <Button
        onClick={handleRemoveClick}
        kind="tertiary"
        color="neutral"
        size="small"
        title="Remove file"
      >
        <RemoveIcon />
      </Button>
      
      <ConfirmationModal
        isOpen={showRemoveModal}
        onClose={() => setShowRemoveModal(false)}
        onConfirm={handleConfirmRemove}
        title="Remove File"
        message={`Are you sure you want to remove "${uploadFile.file.name}" from this collection?`}
        confirmText="Remove"
        confirmColor="danger"
      />
    </Flex>
  );
};

const FileDetails = ({ uploadFile }: { uploadFile: UploadFile }) => {
  const { formatFileSize } = useFileUtils();
  
  return (
    <Flex align="center" gap="density-sm">
      <Text kind="body/regular/sm" style={{ color: '#cccccc' }}>
        {formatFileSize(uploadFile.file.size)}
      </Text>
      <FileStatus uploadFile={uploadFile} />
    </Flex>
  );
};

export const FileItem = ({ uploadFile, onRemove }: FileItemProps) => (
  <Flex 
    gap="density-md"
  >
    <Block style={{ flexShrink: 0 }}>
      <FileIcon uploadFile={uploadFile} />
    </Block>

    <Block style={{ flex: 1, minWidth: 0 }}>
      <FileHeader uploadFile={uploadFile} onRemove={onRemove} />
      <FileDetails uploadFile={uploadFile} />
      <FileMetadataForm fileName={uploadFile.file.name} />
    </Block>
  </Flex>
); 
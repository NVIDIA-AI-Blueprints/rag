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
import { useNewCollectionStore } from "../../store/useNewCollectionStore";
import { useCollectionDrawerStore } from "../../store/useCollectionDrawerStore";
import { useCollectionActions } from "../../hooks/useCollectionActions";
import NvidiaUpload from "../files/NvidiaUpload";
import { Button, Stack, Flex, Text, Spinner } from "@kui/react";

const CloseIcon = () => (
  <svg style={{ width: '16px', height: '16px', color: 'var(--text-color-inverse)' }} fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
  </svg>
);

export const UploaderSection = () => {
  const { selectedFiles, hasInvalidFiles } = useNewCollectionStore();
  const { toggleUploader } = useCollectionDrawerStore();
  const { handleUploadDocuments, isUploading } = useCollectionActions();

  const handleCloseUploader = useCallback(() => {
    toggleUploader(false);
    useNewCollectionStore.getState().reset();
  }, [toggleUploader]);

  const handleValidationChange = useCallback((hasInvalidFiles: boolean) => {
    useNewCollectionStore.getState().setHasInvalidFiles(hasInvalidFiles);
  }, []);

  const handleFilesChange = useCallback((files: File[]) => {
    console.log('🟠 UploaderSection: received', files.length, 'files, calling setFiles');
    useNewCollectionStore.getState().setFiles(files);
  }, []);

  return (
    <Stack 
      gap="density-xl" 
      style={{ 
        borderTop: '1px solid var(--border-color-subtle)',
        paddingTop: '24px',
        marginTop: '24px'
      }}
    >
      <Flex justify="between" align="center" style={{ marginBottom: '16px' }}>
        <Flex align="center" gap="density-md">
          <Text kind="body/bold/lg" style={{ color: 'var(--text-color-inverse)' }}>
            Add New Documents
          </Text>
        </Flex>
        <Button
          onClick={handleCloseUploader}
          disabled={isUploading}
          kind="tertiary"
          size="small"
          data-testid="uploader-close-button"
        >
          <CloseIcon />
        </Button>
      </Flex>
      
      <NvidiaUpload
        onFilesChange={handleFilesChange}
        onValidationChange={handleValidationChange}
        acceptedTypes={['.bmp', '.docx', '.html', '.jpeg', '.json', '.md', '.pdf', '.png', '.pptx', '.sh', '.tiff', '.txt', '.mp3', '.wav']}
        maxFileSize={400}
      />
      
      {selectedFiles.length > 0 && (
        <Button
          onClick={handleUploadDocuments}
          disabled={isUploading || hasInvalidFiles}
          kind="primary"
          color="brand"
          size="large"
          style={{ width: '100%' }}
        >
          {isUploading ? (
            <Flex align="center" gap="density-sm">
              <Spinner size="small" aria-label="Uploading files" />
              Uploading {selectedFiles.length} File{selectedFiles.length > 1 ? 's' : ''}...
            </Flex>
          ) : (
            `Upload ${selectedFiles.length} File${selectedFiles.length > 1 ? 's' : ''}`
          )}
        </Button>
      )}
    </Stack>
  );
}; 
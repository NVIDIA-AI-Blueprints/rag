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
import { useCollectionDocuments } from "../../api/useCollectionDocuments";
import { useCollectionDrawerStore } from "../../store/useCollectionDrawerStore";
import { DocumentItem } from "./DocumentItem";
import { 
  Stack, 
  Spinner, 
  StatusMessage, 
  Button,
  Banner, 
  Divider,
  Block
} from "@kui/react";

const DocumentsEmptyIcon = () => (
  <svg 
    style={{ 
      width: '48px', 
      height: '48px',
      stroke: 'var(--text-color-inverse)'
    }}
    fill="none" 
    strokeWidth="1.5" 
    viewBox="0 0 24 24"
  >
    <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 0 0-3.375-3.375h-1.5A1.125 1.125 0 0 1 13.5 7.125v-1.5a3.375 3.375 0 0 0-3.375-3.375H8.25m2.25 0H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 0 0-9-9Z" />
  </svg>
);

export const DocumentsList = () => {
  const { activeCollection } = useCollectionDrawerStore();
  const { data, isLoading, error, refetch } = useCollectionDocuments(activeCollection?.collection_name || "");

  const handleRetry = useCallback(() => {
    refetch();
  }, [refetch]);

  if (isLoading) {
    return (
      <div className="text-white">
        <Spinner description="Loading documents..." />
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-white">
        <Banner
          kind="header"
          status="error"
          slotSubheading="There was an error loading the documents for this collection."
          slotActions={
            <Button kind="secondary" onClick={handleRetry}>
              Retry
            </Button>
          }
        >
          Failed to load documents
        </Banner>
      </div>
    );
  }

  if (!data?.documents?.length) {
    return (
      <div style={{ color: 'var(--text-color-inverse)' }}>
        <StatusMessage
          slotHeading="No documents yet"
          slotSubheading="This collection is empty. Add files using the 'Add Source' button below to get started."
          slotMedia={<DocumentsEmptyIcon />}
          attributes={{
            StatusMessageHeading: {
              style: { color: 'var(--text-color-inverse)' }
            },
            StatusMessageSubheading: {
              style: { color: 'var(--text-color-inverse)' }
            }
          }}
        />
      </div>
    );
  }

  return (
    <Stack gap="density-lg" data-testid="documents-list">
      {data.documents.map((doc, index) => (
        <Block key={doc.document_name}>
          <DocumentItem 
            name={doc.document_name}
            metadata={doc.metadata}
            collectionName={activeCollection?.collection_name || ""}
          />
          {index < data.documents.length - 1 && <Divider style={{ borderColor: 'var(--border-color-interaction-inverse)' }} />}
        </Block>
      ))}
    </Stack>
  );
}; 
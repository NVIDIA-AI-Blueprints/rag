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
  Divider
} from "@kui/react";
import { FileText } from "lucide-react";

export const DocumentsList = () => {
  const { activeCollection } = useCollectionDrawerStore();
  const { data, isLoading, error, refetch } = useCollectionDocuments(activeCollection?.collection_name || "");

  const handleRetry = useCallback(() => {
    refetch();
  }, [refetch]);

  if (isLoading) {
    return <Spinner description="Loading documents..." />;
  }

  if (error) {
    return (
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
    );
  }

  if (!data?.documents?.length) {
    return (
      <StatusMessage
        slotHeading="No documents yet"
        slotSubheading="This collection is empty. Add files using the 'Add Source' button below to get started."
        slotMedia={<FileText size={48} />}
      />
    );
  }

  return (
    <Stack gap="density-lg" data-testid="documents-list">
      {data.documents.map((doc, index) => (
        <Stack key={doc.document_name} gap="density-lg">
          <DocumentItem 
            name={doc.document_name}
            metadata={doc.metadata}
            collectionName={activeCollection?.collection_name || ""}
            documentInfo={doc.document_info}
          />
          {index < data.documents.length - 1 && <Divider />}
        </Stack>
      ))}
    </Stack>
  );
};

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

import { useEffect, useCallback } from "react";
import NvidiaUpload from "../components/files/NvidiaUpload";
import MetadataSchemaEditor from "../components/schema/MetadataSchemaEditor";
import NewCollectionButtons from "../components/collections/NewCollectionButtons";
import { useNewCollectionStore } from "../store/useNewCollectionStore";
import { FormField, Grid, GridItem, PageHeader, Panel, Stack, TextInput } from "@kui/react";

/**
 * New Collection page component for creating collections.
 * 
 * Provides a multi-step interface for collection creation including
 * file upload, metadata schema definition, and collection naming.
 * 
 * @returns The new collection page component
 */
export default function NewCollection() {
  const { collectionName, setCollectionName, setCollectionNameTouched, reset } = useNewCollectionStore();

  useEffect(() => {
    // cleanup when leaving the page
    return () => {
      reset();
    };
  }, [reset]);

  const handleValidationChange = useCallback((hasInvalidFiles: boolean) => {
    const { setHasInvalidFiles } = useNewCollectionStore.getState();
    setHasInvalidFiles(hasInvalidFiles);
  }, []);

  const handleFilesChange = useCallback((files: File[]) => {
    const { setFiles } = useNewCollectionStore.getState();
    setFiles(files);
  }, []);

  return (
    <Grid cols={12} gap="density-lg" padding="density-lg">
      <GridItem cols={12}>
        <PageHeader
          slotHeading="Create New Collection"
          slotSubheading="Upload source files and define metadata schema for this collection."
        />
      </GridItem>
      <GridItem cols={6}>
        <Panel>
          <Stack gap="density-lg">
            <FormField
              slotLabel="Collection Name"
              slotHelp="We will automatically try to validate the collection name."
              required
            >
              <TextInput
                value={collectionName}
                onChange={(e) => setCollectionName(e.target.value.replace(/\s+/g, "_"))}
                onBlur={() => setCollectionNameTouched(true)}
              />
            </FormField>
            <MetadataSchemaEditor />
          </Stack>
        </Panel>
      </GridItem>
      <GridItem cols={6}>
        <Panel>
          <Stack 
            gap="density-xl"
            style={{ 
              borderTop: '1px solid var(--border-color-subtle)',
            }}
          >
            <NvidiaUpload 
              onFilesChange={handleFilesChange}
              onValidationChange={handleValidationChange}
              acceptedTypes={['.bmp', '.docx', '.html', '.jpeg', '.json', '.md', '.pdf', '.png', '.pptx', '.sh', '.tiff', '.txt', '.mp3', '.wav']}
              maxFileSize={400}
            />
          </Stack>
        </Panel>
      </GridItem>
      <GridItem cols={12}>
        <NewCollectionButtons />
      </GridItem>
    </Grid>
  );
}
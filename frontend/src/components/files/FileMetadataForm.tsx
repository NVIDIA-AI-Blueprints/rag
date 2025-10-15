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
import { MetadataField } from "./MetadataField";
import type { UIMetadataField } from "../../types/collections";

interface FileMetadataFormProps {
  fileName: string;
}

export const FileMetadataForm = ({ fileName }: FileMetadataFormProps) => {
  const { metadataSchema, fileMetadata, updateMetadataField } = useNewCollectionStore();

  const handleFieldChange = useCallback((fieldName: string, value: unknown) => {
    updateMetadataField(fileName, fieldName, value);
  }, [fileName, updateMetadataField]);

  if (metadataSchema.length === 0) {
    return null;
  }

  return (
    <div className="mt-4 pt-4 border-t border-neutral-700">
      <div className="space-y-4">
        {metadataSchema
          .filter((field: UIMetadataField) => field.name !== 'filename') // Filter out filename field
          .map((field: UIMetadataField) => (
          <MetadataField
            key={field.name}
            fileName={fileName}
            field={field}
            value={(() => {
              const existingValue = fileMetadata[fileName]?.[field.name];
              if (existingValue !== undefined) return existingValue;
              switch (field.type) {
                case "boolean": return false;
                case "array": return [];
                case "integer":
                case "float":
                case "number": return null;
                default: return "";
              }
            })()}
            onChange={handleFieldChange}
          />
        ))}
      </div>
    </div>
  );
}; 
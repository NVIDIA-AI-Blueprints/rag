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
import { useSchemaEditor } from "../../hooks/useSchemaEditor";
import { FieldEditForm } from "./FieldEditForm";
import { FieldDisplayCard } from "./FieldDisplayCard";
import type { UIMetadataField } from "../../types/collections";

export const FieldsList = () => {
  const {
    metadataSchema,
    editingIndex,
    editValues,
    startEditing,
    updateEditValue,
    commitUpdate,
    cancelEdit,
    deleteField,
  } = useSchemaEditor();

  const handleUpdate = useCallback((idx: number) => (updates: Partial<UIMetadataField>) => {
    updateEditValue(idx, updates);
  }, [updateEditValue]);

  const handleSave = useCallback((idx: number) => () => {
    commitUpdate(idx);
  }, [commitUpdate]);

  const handleEdit = useCallback((idx: number) => () => {
    startEditing(idx);
  }, [startEditing]);

  const handleDelete = useCallback((idx: number) => () => {
    deleteField(idx);
  }, [deleteField]);

  return (
    <div className="space-y-3">
      {metadataSchema.map((field, idx) =>
        editingIndex === idx ? (
          <FieldEditForm
            key={idx}
            field={editValues[idx] || field}
            onUpdate={handleUpdate(idx)}
            onSave={handleSave(idx)}
            onCancel={cancelEdit}
          />
        ) : (
          <FieldDisplayCard
            key={idx}
            field={field}
            onEdit={handleEdit(idx)}
            onDelete={handleDelete(idx)}
          />
        )
      )}
    </div>
  );
}; 
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
import { useNewCollectionStore } from "../store/useNewCollectionStore";
import type { UIMetadataField } from "../types/collections";

export const useSchemaEditor = () => {
  const { metadataSchema, setMetadataSchema } = useNewCollectionStore();
  const [showSchemaEditor, setShowSchemaEditor] = useState(true);
  const [editingIndex, setEditingIndex] = useState<number | null>(null);
  const [editValues, setEditValues] = useState<Record<number, UIMetadataField>>({});
  const [fieldNameError, setFieldNameError] = useState<string | null>(null);

  const toggleEditor = useCallback(() => {
    setShowSchemaEditor(prev => !prev);
  }, []);

  const startEditing = useCallback((idx: number) => {
    setEditingIndex(idx);
    setEditValues(prev => ({
      ...prev,
      [idx]: {
        // Copy all properties from the existing field
        ...metadataSchema[idx],
        // Ensure backward compatibility with optional field
        optional: metadataSchema[idx].optional ?? !metadataSchema[idx].required,
      },
    }));
  }, [metadataSchema]);

  const updateEditValue = useCallback((idx: number, updatedValue: Partial<UIMetadataField>) => {
    setEditValues(prev => ({
      ...prev,
      [idx]: {
        ...prev[idx],
        ...updatedValue,
      },
    }));
  }, []);

  const commitUpdate = useCallback((index: number) => {
    const editValue = editValues[index];
    if (!editValue) return;
    
    const updated = [...metadataSchema];
    updated[index] = {
      ...editValue,
      name: editValue.name.trim(),
    };
    setMetadataSchema(updated);
    setEditingIndex(null);
    setEditValues(prev => {
      const copy = { ...prev };
      delete copy[index];
      return copy;
    });
  }, [editValues, metadataSchema, setMetadataSchema]);

  const cancelEdit = useCallback(() => {
    setEditingIndex(null);
  }, []);

  const deleteField = useCallback((index: number) => {
    const updated = metadataSchema.filter((_, i) => i !== index);
    setMetadataSchema(updated);
    setEditingIndex(null);
    setEditValues(prev => {
      const copy = { ...prev };
      delete copy[index];
      return copy;
    });
  }, [metadataSchema, setMetadataSchema]);

  return {
    metadataSchema,
    showSchemaEditor,
    editingIndex,
    editValues,
    fieldNameError,
    setFieldNameError,
    toggleEditor,
    startEditing,
    updateEditValue,
    commitUpdate,
    cancelEdit,
    deleteField,
  };
}; 
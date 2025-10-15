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

export const useNewFieldForm = () => {
  const { metadataSchema, setMetadataSchema } = useNewCollectionStore();
  const [newField, setNewField] = useState<UIMetadataField>({
    name: "",
    type: "string",
    required: false,
  });
  const [fieldNameError, setFieldNameError] = useState<string | null>(null);

  const updateNewField = useCallback((updates: Partial<UIMetadataField>) => {
    setNewField(prev => {
      const updated = { ...prev, ...updates };
      
      // Validation: array_type is required for array fields
      if (updated.type === "array" && !updated.array_type) {
        updated.array_type = "string";
      }
      
      // Clear array_type if not an array field
      if (updated.type !== "array") {
        delete updated.array_type;
      }
      
      return updated;
    });
  }, []);

  const resetForm = useCallback(() => {
    setNewField({ 
      name: "", 
      type: "string", 
      required: false 
    });
    setFieldNameError(null);
  }, []);

  const validateFieldName = useCallback((name: string): string | null => {
    const trimmed = name.trim();
    
    // Check if empty
    if (!trimmed) {
      return "Field name is required";
    }
    
    // Check for invalid characters or patterns
    if (!/^[a-zA-Z][a-zA-Z0-9_]*$/.test(trimmed)) {
      return "Field name must start with a letter and contain only letters, numbers, and underscores";
    }
    
    // Check for reserved names
    const reservedNames = ["filename", "id", "_id", "content"];
    if (reservedNames.includes(trimmed.toLowerCase())) {
      return `"${trimmed}" is a reserved field name`;
    }
    
    // Check for duplicate names (case-insensitive)
    if (metadataSchema.some(f => f.name.toLowerCase() === trimmed.toLowerCase())) {
      return "Field name already exists";
    }
    
    return null;
  }, [metadataSchema]);

  const validateField = useCallback((field: UIMetadataField): string | null => {
    // Validate field name
    const nameError = validateFieldName(field.name);
    if (nameError) return nameError;
    
    // Validate array fields have array_type
    if (field.type === "array" && !field.array_type) {
      return "Array fields must specify an element type";
    }
    
    // Validate max_length
    if (field.max_length !== undefined) {
      if (field.max_length <= 0) {
        return "Maximum length must be greater than 0";
      }
      if (!["string", "array"].includes(field.type)) {
        return "Maximum length is only applicable to string and array fields";
      }
    }
    
    return null;
  }, [validateFieldName]);

  const validateAndAdd = useCallback(() => {
    const trimmed = newField.name.trim();
    const fieldToAdd = { ...newField, name: trimmed };
    
    const error = validateField(fieldToAdd);
    if (error) {
      setFieldNameError(error);
      return false;
    }

    setFieldNameError(null);
    
    // Convert to final format (maintain backward compatibility)
    const finalField: UIMetadataField = {
      ...fieldToAdd,
      // Keep optional for backward compatibility 
      optional: !fieldToAdd.required,
    };
    
    const updated = [...metadataSchema, finalField];
    setMetadataSchema(updated);
    resetForm();
    return true;
  }, [newField, validateField, metadataSchema, setMetadataSchema, resetForm]);

  // Enhanced validation for canAdd
  const canAdd = useCallback(() => {
    const trimmed = newField.name.trim();
    if (!trimmed) return false;
    
    // Quick validation without setting errors
    const error = validateField({ ...newField, name: trimmed });
    return !error;
  }, [newField, validateField]);

  return {
    newField,
    fieldNameError,
    updateNewField,
    validateAndAdd,
    resetForm,
    canAdd: canAdd(),
    validateFieldName,
  };
}; 
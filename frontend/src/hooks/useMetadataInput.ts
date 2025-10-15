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
import { useNewCollectionStore } from "../store/useNewCollectionStore";
import type { MetadataFieldType } from "../types/collections";

/**
 * Custom hook for handling metadata input validation and processing.
 * 
 * Provides validation, parsing, and formatting functions for different metadata
 * field types including strings, numbers, booleans, dates, and arrays. Handles
 * value conversion and updates to the collection store.
 * 
 * @returns Object with validation and handling functions for metadata inputs
 * 
 * @example
 * ```tsx
 * const { handleInputChange, validateValue, parseValue } = useMetadataInput();
 * handleInputChange("tags", "tag1, tag2, tag3", "array");
 * ```
 */
export const useMetadataInput = () => {
  const { updateMetadataField } = useNewCollectionStore();

  const validateValue = useCallback((value: unknown, fieldType: MetadataFieldType): unknown => {
    if (value === null || value === undefined) return value;

    // Convert to string for validation if it's not already processed
    const stringValue = typeof value === 'string' ? value : String(value);
    
    if (!stringValue) return stringValue;

    switch (fieldType) {
      case "datetime":
        // Fix datetime format if needed (add seconds if missing)
        if (stringValue.length === 16) {
          return `${stringValue}:00`;
        }
        return stringValue;
      
      case "integer": {
        // Return actual integer, not string
        if (typeof value === 'number' && Number.isInteger(value)) {
          return value;
        }
        const intVal = parseInt(stringValue);
        if (isNaN(intVal)) {
          throw new Error(`Invalid integer value: ${stringValue}`);
        }
        return intVal;
      }
      
      case "float":
      case "number": {
        // Return actual number, not string
        if (typeof value === 'number') {
          return value;
        }
        const floatVal = parseFloat(stringValue);
        if (isNaN(floatVal)) {
          throw new Error(`Invalid number value: ${stringValue}`);
        }
        return floatVal;
      }
      
      case "boolean": {
        // Return actual boolean, not string
        if (typeof value === 'boolean') {
          return value;
        }
        const lowerValue = stringValue.toLowerCase();
        if (["true", "1", "yes", "on"].includes(lowerValue)) {
          return true;
        } else if (["false", "0", "no", "off"].includes(lowerValue)) {
          return false;
        } else {
          throw new Error(`Invalid boolean value: ${stringValue}. Use true/false, 1/0, yes/no, or on/off`);
        }
      }
      
      case "array":
        // Return actual array, not JSON string
        if (Array.isArray(value)) {
          return value;
        }
        try {
          return JSON.parse(stringValue);
        } catch {
          throw new Error(`Invalid array value: ${stringValue}`);
        }
      
      case "string":
      default:
        return stringValue;
    }
  }, []);

  const handleFieldChange = useCallback((
    fileName: string, 
    fieldName: string, 
    value: unknown, 
    fieldType: MetadataFieldType
  ) => {
    try {
      const processedValue = validateValue(value, fieldType);
      updateMetadataField(fileName, fieldName, processedValue);
    } catch (error) {
      console.warn(`Metadata validation error for ${fieldName}:`, error);
      // Still update with the raw value to allow user to see and fix
      updateMetadataField(fileName, fieldName, value);
    }
  }, [updateMetadataField, validateValue]);

  const createChangeHandler = useCallback((
    fileName: string, 
    fieldName: string, 
    fieldType: MetadataFieldType
  ) => (e: React.ChangeEvent<HTMLInputElement>) => {
    let value: unknown = e.target.value;
    
    // Handle checkbox inputs for boolean fields
    if (fieldType === "boolean" && e.target.type === "checkbox") {
      value = e.target.checked;
    }
    
    handleFieldChange(fileName, fieldName, value, fieldType);
  }, [handleFieldChange]);

  const validateFieldValue = useCallback((
    value: unknown, 
    fieldType: MetadataFieldType, 
    required: boolean = false,
    maxLength?: number
  ): { isValid: boolean; error?: string } => {
    const stringValue = typeof value === 'string' ? value : String(value || '');
    
    // Check required fields - for boolean fields, "false" is a valid value
    if (required) {
      if (fieldType === "boolean") {
        // For boolean fields, check if we have a boolean value or a valid string representation
        if (typeof value === 'boolean') {
          return { isValid: true };
        }
        if (typeof value === 'string') {
          const lowerValue = value.toLowerCase().trim();
          const validBooleanValues = ["true", "false", "1", "0", "yes", "no", "on", "off"];
          if (!lowerValue || !validBooleanValues.includes(lowerValue)) {
            return { isValid: false, error: "This field is required" };
          }
        }
      } else {
        // For non-boolean fields, check if value is empty
        if (!stringValue.trim()) {
          return { isValid: false, error: "This field is required" };
        }
      }
    }

    // Skip validation for empty optional fields
    if (!stringValue.trim() && !required) {
      return { isValid: true };
    }

    // Check max length for strings
    if (fieldType === "string" && maxLength && stringValue.length > maxLength) {
      return { isValid: false, error: `Maximum length is ${maxLength} characters` };
    }

    // Check array max length
    if (fieldType === "array" && maxLength) {
      let arrayValue: unknown;
      if (Array.isArray(value)) {
        arrayValue = value;
      } else {
        try {
          arrayValue = JSON.parse(stringValue);
        } catch {
          return { isValid: false, error: "Invalid array format" };
        }
      }
      
      if (Array.isArray(arrayValue) && arrayValue.length > maxLength) {
        return { isValid: false, error: `Maximum ${maxLength} items allowed` };
      }
    }

    try {
      validateValue(value, fieldType);
      return { isValid: true };
    } catch (error) {
      return { isValid: false, error: (error as Error).message };
    }
  }, [validateValue]);

  return {
    handleFieldChange,
    createChangeHandler,
    validateFieldValue,
    validateValue,
  };
}; 
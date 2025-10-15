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

/**
 * Interface representing a form field's state and validation status.
 */
interface FormField {
  value: unknown;
  stringValue: string;
  isValid: boolean;
  hasChanged: boolean;
}

/**
 * Custom hook for managing form validation state and field tracking.
 * 
 * Provides comprehensive form validation functionality including field state
 * management, validation tracking, and form submission validation. Handles
 * field value changes, validation status, and form-wide validation state.
 * 
 * @returns Object with form field management and validation functions
 * 
 * @example
 * ```tsx
 * const { fields, setField, validateField, isFormValid } = useFormValidation();
 * setField('email', 'user@example.com', true);
 * ```
 */
export const useFormValidation = () => {
  const [fields, setFields] = useState<Record<string, FormField>>({});

  const registerField = useCallback((name: string, initialValue: unknown) => {
    const stringValue = Array.isArray(initialValue) 
      ? initialValue.join(", ") 
      : (initialValue != null ? String(initialValue) : "");
    
    setFields(prev => ({
      ...prev,
      [name]: {
        value: initialValue,
        stringValue,
        isValid: true,
        hasChanged: false,
      }
    }));
  }, []);

  const updateField = useCallback((name: string, stringValue: string, parser?: (value: string) => unknown) => {
    setFields(prev => {
      const field = prev[name];
      if (!field) return prev;

      let parsedValue: unknown = stringValue;
      let isValid = true;

      if (parser) {
        try {
          parsedValue = parser(stringValue);
          // Additional validation can be added here
        } catch {
          isValid = false;
        }
      }

      return {
        ...prev,
        [name]: {
          value: parsedValue,
          stringValue,
          isValid,
          hasChanged: true,
        }
      };
    });
  }, []);

  const syncField = useCallback((name: string, externalValue: unknown) => {
    setFields(prev => {
      const field = prev[name];
      if (!field) return prev;

      const stringValue = Array.isArray(externalValue) 
        ? externalValue.join(", ") 
        : (externalValue != null ? String(externalValue) : "");

      return {
        ...prev,
        [name]: {
          ...field,
          value: externalValue,
          stringValue,
          hasChanged: false,
        }
      };
    });
  }, []);

  const getField = useCallback((name: string) => fields[name], [fields]);

  const isFieldValid = useCallback((name: string) => {
    return fields[name]?.isValid ?? true;
  }, [fields]);

  return {
    registerField,
    updateField,
    syncField,
    getField,
    isFieldValid,
  };
}; 
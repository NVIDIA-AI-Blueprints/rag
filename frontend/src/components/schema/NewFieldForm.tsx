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

import { useCallback, useState } from "react";
import { useNewFieldForm } from "../../hooks/useNewFieldForm";
import type { MetadataFieldType, ArrayElementType } from "../../types/collections";
import { 
  Button, 
  TextInput, 
  Select, 
  TextArea, 
  Checkbox, 
  FormField, 
  Stack, 
  Block, 
  Text, 
  Flex 
} from "@kui/react";

const AddIcon = () => (
  <svg className="w-4 h-4 text-[var(--nv-green)]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
  </svg>
);

const FIELD_TYPES: MetadataFieldType[] = [
  "string", "integer", "float", "number", "boolean", "datetime", "array"
];

const ARRAY_TYPES: ArrayElementType[] = [
  "string", "number", "integer", "float" //, "boolean"
];

export const NewFieldForm = () => {

  const { newField, fieldNameError, updateNewField, validateAndAdd, canAdd } = useNewFieldForm();
  const [showAdvanced, setShowAdvanced] = useState(false);

  const handleNameChange = useCallback((value: string) => {
    updateNewField({ name: value });
  }, [updateNewField]);

  const handleRequiredChange = useCallback((checked: string | boolean) => {
    updateNewField({ required: !!checked });
  }, [updateNewField]);

  const handleDescriptionChange = useCallback((value: string) => {
    updateNewField({ description: value || undefined });
  }, [updateNewField]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      validateAndAdd();
    }
  }, [validateAndAdd]);

  return (
    <Block paddingY="6" style={{ borderTop: '1px solid var(--border-color-subtle)' }}>
      <Stack gap="4">
        <Stack gap="1">
          <Flex align="center" gap="2">
            <AddIcon />
            <Text kind="body/bold/md">Add New Field</Text>
          </Flex>
          <Text kind="body/regular/xs" style={{ color: 'var(--text-color-subtle)' }}>
            Create a metadata field for this collection
          </Text>
        </Stack>
        
        <Block 
          padding="4" 
          style={{ 
            backgroundColor: 'var(--background-color-surface-raised)',
            border: '1px solid var(--border-color-subtle)',
            borderRadius: '8px'
          }}
        >
          <Stack gap="4">
            {/* Basic Fields */}
            <Flex gap="4" wrap="wrap">
              {/* Field Name */}
              <Block style={{ flex: 1 }}>
                <FormField
                  slotLabel="Field Name"
                  status={fieldNameError ? "error" : undefined}
                  slotHelp={fieldNameError}
                >
                  <TextInput
                    placeholder="e.g. category, author, department"
                    value={newField.name}
                    onValueChange={handleNameChange}
                    onKeyDown={handleKeyDown}
                    status={fieldNameError ? "error" : undefined}
                  />
                </FormField>
              </Block>

              {/* Field Type */}
              <Block style={{ flex: 1 }}>
                <FormField
                  slotLabel="Field Type"
                >
                  <Select
                    value={newField.type}
                    items={FIELD_TYPES}
                    onValueChange={(value) => {
                      const newType = value as MetadataFieldType;
                      updateNewField({ 
                        type: newType,
                        array_type: newType === "array" ? "string" : undefined
                      });
                    }}
                  />
                </FormField>
              </Block>
            </Flex>

            {/* Array Type Selection (only for array fields) */}
            {newField.type === "array" && (
              <FormField
                slotLabel="Array Element Type"
                required
                slotHelp="The data type for elements stored in this array"
              >
                <Select
                  value={newField.array_type || ""}
                  placeholder="Select element type"
                  items={ARRAY_TYPES}
                  onValueChange={(value) => {
                    updateNewField({ array_type: value as ArrayElementType });
                  }}
                />
              </FormField>
            )}

            {/* Advanced Options Toggle */}
            <Button
              kind="tertiary"
              size="small"
              onClick={() => setShowAdvanced(!showAdvanced)}
            >
              <svg
                style={{ 
                  width: '12px', 
                  height: '12px',
                  transform: showAdvanced ? 'rotate(90deg)' : 'rotate(0deg)',
                  transition: 'transform 0.2s',
                  marginRight: '4px'
                }}
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
              Advanced Options
            </Button>

            {/* Advanced Fields */}
            {showAdvanced && (
              <Block 
                padding="6" 
                style={{ 
                  backgroundColor: 'var(--background-color-surface-base)',
                  border: '1px solid var(--border-color-subtle)',
                  borderRadius: '8px'
                }}
              >
                <Stack gap="6">
                  {/* Max Length and Required Field side by side */}
                  <Flex gap="4" wrap="wrap" align="center">
                    {/* Max Length */}
                    {(newField.type === "string" || newField.type === "array") && (
                      <Block style={{ flex: 1 }}>
                        <FormField
                          slotLabel={`Maximum ${newField.type === "string" ? "Length" : "Items"}`}
                        >
                          <TextInput
                            type="number"
                            min="1"
                            value={newField.max_length?.toString() || ""}
                            onValueChange={(value) => {
                              updateNewField({ 
                                max_length: value ? parseInt(value) : undefined 
                              });
                            }}
                            placeholder={newField.type === "string" ? "e.g. 100" : "e.g. 10"}
                          />
                        </FormField>
                      </Block>
                    )}

                    {/* Required Field */}
                    <Block paddingBottom="1">
                      <Checkbox
                        checked={newField.required}
                        onCheckedChange={handleRequiredChange}
                        slotLabel="Required field"
                      />
                    </Block>
                  </Flex>

                  {/* Description */}
                  <FormField slotLabel="Description">
                    <TextArea
                      value={newField.description || ""}
                      onValueChange={handleDescriptionChange}
                      onKeyDown={handleKeyDown}
                      rows={3}
                      placeholder="Optional description for this field"
                    />
                  </FormField>
                </Stack>
              </Block>
            )}

            {/* Add Button */}
            <Button
              onClick={validateAndAdd}
              disabled={!canAdd}
              style={{ width: '100%' }}
            >
              Add Field
            </Button>
          </Stack>
        </Block>
      </Stack>
    </Block>
  );
}; 
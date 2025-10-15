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
import type { UIMetadataField, MetadataFieldType, ArrayElementType } from "../../types/collections";
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

interface FieldEditFormProps {
  field: UIMetadataField;
  onUpdate: (updates: Partial<UIMetadataField>) => void;
  onSave: () => void;
  onCancel: () => void;
}

const EditIcon = () => (
  <svg className="w-4 h-4 text-[var(--nv-green)]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
  </svg>
);

const FIELD_TYPES: MetadataFieldType[] = [
  "string", "integer", "float", "number", "boolean", "datetime", "array"
];

const ARRAY_TYPES: ArrayElementType[] = [
  "string", "number", "integer", "float", "boolean"
];

export const FieldEditForm = ({ field, onUpdate, onSave, onCancel }: FieldEditFormProps) => {
  const [showAdvanced, setShowAdvanced] = useState(false);

  const handleNameChange = useCallback((value: string) => {
    onUpdate({ name: value });
  }, [onUpdate]);

  const handleRequiredChange = useCallback((checked: string | boolean) => {
    onUpdate({ required: !!checked });
  }, [onUpdate]);

  const handleDescriptionChange = useCallback((value: string) => {
    onUpdate({ description: value || undefined });
  }, [onUpdate]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      onSave();
    }
  }, [onSave]);

  return (
    <Block padding="5">
      <Stack gap="4">
        <Flex align="center" gap="2">
          <EditIcon />
          <Text kind="body/bold/xs">
            Editing Field
          </Text>
        </Flex>
        
        {/* Basic Fields */}
        <Flex gap="4" wrap="wrap">
          {/* Field Name */}
          <Block style={{ flex: 1 }}>
            <FormField
              slotLabel="Field Name"
              required
            >
              <TextInput
                autoFocus
                value={field.name || ""}
                onValueChange={handleNameChange}
                onKeyDown={handleKeyDown}
              />
            </FormField>
          </Block>

          {/* Field Type */}
          <Block style={{ flex: 1 }}>
            <FormField
              slotLabel="Field Type"
              required
            >
              <Select
                value={field.type || "string"}
                items={FIELD_TYPES}
                onValueChange={(value) => {
                  const newType = value as MetadataFieldType;
                  onUpdate({ 
                    type: newType,
                    array_type: newType === "array" ? (field.array_type || "string") : undefined
                  });
                }}
              />
            </FormField>
          </Block>
        </Flex>

        {/* Array Type Selection (only for array fields) */}
        {field.type === "array" && (
          <FormField
            slotLabel="Array Element Type"
            required
            slotHelp="The data type for elements stored in this array"
          >
            <Select
              value={field.array_type || ""}
              placeholder="Select element type"
              items={ARRAY_TYPES}
              onValueChange={(value) => {
                onUpdate({ array_type: value as ArrayElementType });
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
            padding="4" 
            style={{ 
              backgroundColor: 'var(--background-color-surface-base)',
              border: '1px solid var(--border-color-subtle)',
              borderRadius: '8px'
            }}
          >
            <Stack gap="4">
              {/* Max Length and Required Field side by side */}
              <Flex gap="4" wrap="wrap" align="center">
                {/* Max Length */}
                {(field.type === "string" || field.type === "array") && (
                  <Block style={{ flex: 1 }}>
                    <FormField
                      slotLabel={`Maximum ${field.type === "string" ? "Length" : "Items"}`}
                    >
                      <TextInput
                        type="number"
                        min="1"
                        value={field.max_length?.toString() || ""}
                        onValueChange={(value) => {
                          onUpdate({ 
                            max_length: value ? parseInt(value) : undefined 
                          });
                        }}
                        placeholder={field.type === "string" ? "e.g. 100" : "e.g. 10"}
                      />
                    </FormField>
                  </Block>
                )}

                {/* Required Field */}
                <Block paddingBottom="1">
                  <Checkbox
                    checked={field.required || false}
                    onCheckedChange={handleRequiredChange}
                    slotLabel="Required field"
                  />
                </Block>
              </Flex>

              {/* Description */}
              <FormField slotLabel="Description">
                <TextArea
                  value={field.description || ""}
                  onValueChange={handleDescriptionChange}
                  onKeyDown={handleKeyDown}
                  rows={2}
                  placeholder="Optional description for this field"
                />
              </FormField>
            </Stack>
          </Block>
        )}

        {/* Save/Cancel Buttons */}
        <Flex justify="end" gap="3" style={{ paddingTop: '8px' }}>
          <Button
            kind="secondary"
            onClick={onCancel}
          >
            Cancel
          </Button>
          <Button
            onClick={onSave}
          >
            Save Changes
          </Button>
        </Flex>
      </Stack>
    </Block>
  );
}; 
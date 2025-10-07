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

import { useCallback } from 'react';
import { Switch, FormField, TextInput, TextArea, Flex, Text, Slider } from "@kui/react";

/**
 * Props for the SettingToggle component.
 */
interface SettingToggleProps {
  label: string;
  description?: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
}

/**
 * Props for the SettingSlider component.
 */
interface SettingSliderProps {
  label: string;
  description?: string;
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
  step: number;
  'aria-label'?: string;
  'data-testid'?: string;
  disabled?: boolean;
}

/**
 * Props for the SettingInput component.
 */
interface SettingInputProps {
  label: string;
  description?: string;
  value: string;
  onChange: (value: string) => void;
  type?: "text" | "number";
  placeholder?: string;
  min?: number;
  max?: number;
  isValid?: boolean;
  validationMessage?: string;
}

/**
 * A toggle switch component for settings using KUI Switch.
 */
export const SettingToggle = ({ 
  label, 
  description, 
  checked, 
  onChange 
}: SettingToggleProps) => {
  return (
    <FormField
      slotLabel={label}
      slotHelp={description}
    >
      {(args) => (
        <Switch
          {...args}
          slotLabel={label}
          checked={checked}
          onCheckedChange={onChange}
        />
      )}
    </FormField>
  );
};

/**
 * A slider component for numeric settings using KUI Slider and FormField.
 */
export const SettingSlider = ({ 
  label, 
  description, 
  value, 
  onChange, 
  min, 
  max, 
  step,
  'aria-label': ariaLabel,
  'data-testid': dataTestId,
  disabled
}: SettingSliderProps) => {
  const handleClick = () => {
    if (disabled) {
      // Activate by setting to current display value
      onChange(value);
    }
  };

  return (
    <FormField
      slotLabel={label}
      slotHelp={description}
    >
      {(args) => (
        <Flex align="center" gap="3" onClick={handleClick} style={{ flex: 1, cursor: disabled ? 'pointer' : 'default' }}>
            <Slider
              {...args}
              value={value}
              onValueChange={(newValue) => onChange(newValue)}
              min={min}
              max={max}
              step={step}
              style={{ flex: 1 }}
              aria-label={ariaLabel || label}
              data-testid={dataTestId}
              disabled={disabled}
            />
          {!disabled ? (
            <Text kind="body/regular/sm" style={{ minWidth: '3rem', textAlign: 'right' }}>
              {value}
            </Text>
          ) : (
            <Text kind="body/regular/sm" style={{ minWidth: '3rem', textAlign: 'right', color: 'var(--text-secondary)' }}>
              Default value - click to change
            </Text>
          )}
        </Flex>
      )}
    </FormField>
  );
};

/**
 * An input component for settings using KUI TextInput and FormField.
 */
export const SettingInput = ({ 
  label, 
  description, 
  value, 
  onChange, 
  type = "text", 
  placeholder,
  min,
  max,
  isValid = true,
  validationMessage
}: SettingInputProps) => {
  const handleChange = useCallback((newValue: string) => {
    onChange(newValue);
  }, [onChange]);

  return (
    <FormField
      slotLabel={label}
      slotHelp={description}
      slotError={!isValid ? validationMessage : undefined}
      status={!isValid ? "error" : undefined}
    >
      {(args) => (
        <TextInput
          {...args}
          type={type}
          value={value}
          onValueChange={handleChange}
          placeholder={placeholder}
          min={min}
          max={max}
        />
      )}
    </FormField>
  );
};

/**
 * A text area component for settings using KUI TextArea and FormField.
 */
export const SettingTextInput = ({ 
  label, 
  description, 
  value,
  onChange, 
  placeholder 
}: Omit<SettingInputProps, 'type' | 'min' | 'max' | 'isValid' | 'validationMessage'>) => {
  const handleChange = useCallback((newValue: string) => {
    onChange(newValue);
  }, [onChange]);

  return (
    <FormField
      slotLabel={label}
      slotHelp={description}
    >
      {(args) => (
        <TextArea
          {...args}
          value={value}
          onValueChange={handleChange}
          placeholder={placeholder}
          rows={3}
        />
      )}
    </FormField>
  );
}; 
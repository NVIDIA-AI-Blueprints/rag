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

import { FormField, TextInput, Switch, Stack, Flex, Text, Button, Tag } from "@kui/react";
import { useSettingsStore } from "../../store/useSettingsStore";
import { useState } from "react";

export const AdvancedSection = () => {
  const { useLocalStorage, stopTokens, set: setSettings } = useSettingsStore();
  const [newTokenInput, setNewTokenInput] = useState("");

  const handleAddToken = () => {
    const token = newTokenInput.trim();
    if (token && !(stopTokens || []).includes(token)) {
      setSettings({ stopTokens: [...(stopTokens || []), token] });
      setNewTokenInput("");
    }
  };

  const handleRemoveToken = (tokenToRemove: string) => {
    setSettings({ 
      stopTokens: (stopTokens || []).filter(token => token !== tokenToRemove) 
    });
  };

  const handleTokenInputChange = (value: string) => {
    setNewTokenInput(value);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleAddToken();
    }
  };

  return (
    <Stack gap="density-lg">
      {/* Theme Toggle */}
      {/* <FormField
        slotLabel="Theme"
        slotHelp="Choose between light and dark theme for the application interface."
      >
        {(args) => (
          <Flex align="center" gap="density-sm">
            <Switch
              {...args}
              checked={isDark}
              onCheckedChange={toggleTheme}
            />
            <Text kind="body/regular/sm">
              {isDark ? 'Dark Theme' : 'Light Theme'}
            </Text>
          </Flex>
        )}
      </FormField> */}

      {/* Local Storage Toggle */}
      <FormField
        slotLabel="Use Local Storage"
        slotHelp="Enable to persist settings and data in your browser's local storage. When disabled, all local storage will be cleared."
      >
        {(args) => (
          <Flex align="center" gap="density-sm">
            <Switch
              {...args}
              checked={useLocalStorage ?? false}
              onCheckedChange={(checked) => setSettings({ useLocalStorage: checked })}
            />
            <Text kind="body/regular/sm">
              {useLocalStorage ? 'Enabled' : 'Disabled'}
            </Text>
          </Flex>
        )}
      </FormField>

      {/* Stop Tokens */}
      <FormField
        slotLabel="Stop Tokens"
        slotHelp="Add text tokens that will stop text generation when encountered. Click on a token to remove it."
      >
        <Stack gap="density-md">
          {/* Input field to add new tokens */}
          <Flex gap="density-sm" align="center">
            <TextInput
              value={newTokenInput}
              onValueChange={handleTokenInputChange}
              onKeyDown={handleKeyPress}
              placeholder="Enter stop token"
              style={{ flex: 1 }}
            />
            <Button 
              onClick={handleAddToken}
              disabled={!newTokenInput.trim()}
              kind="tertiary"
              color="brand"
              size="small"
            >
              Add
            </Button>
          </Flex>
          
          {/* Display existing tokens */}
          {stopTokens && stopTokens.length > 0 && (
            <Flex gap="density-sm" style={{ flexWrap: 'wrap' }}>
              {stopTokens.map((token, index) => (
                <Tag
                  key={`${token}-${index}`}
                  color="gray"
                  kind="outline"
                  density="compact"
                  onClick={() => handleRemoveToken(token)}
                  style={{ cursor: 'pointer' }}
                  title="Click to remove"
                >
                  <Flex align="center" gap="density-xs">
                    <Text kind="body/regular/sm">{token}</Text>
                    <svg 
                      style={{ width: '12px', height: '12px' }} 
                      fill="none" 
                      stroke="currentColor" 
                      strokeWidth="2" 
                      viewBox="0 0 24 24"
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </Flex>
                </Tag>
              ))}
            </Flex>
          )}
        </Stack>
      </FormField>
    </Stack>
  );
}; 
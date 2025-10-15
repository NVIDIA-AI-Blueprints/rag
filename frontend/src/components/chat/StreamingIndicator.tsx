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

import { Flex, Text } from "@kui/react";

interface StreamingIndicatorProps {
  text?: string;
}

export const StreamingIndicator = ({ 
  text 
}: StreamingIndicatorProps) => (
  <Flex align="center" gap="2">
    {text && (
      <Text kind="body/regular/sm" style={{ color: 'var(--text-color-subtle)' }}>
        {text}
      </Text>
    )}
    <Flex gap="1">
      <div 
        style={{
          width: '6px',
          height: '6px',
          backgroundColor: 'var(--color-brand)',
          borderRadius: '50%',
          animation: 'bounce 1.4s infinite',
          animationDelay: '-0.32s'
        }}
      />
      <div 
        style={{
          width: '6px',
          height: '6px',
          backgroundColor: 'var(--color-brand)',
          borderRadius: '50%',
          animation: 'bounce 1.4s infinite',
          animationDelay: '-0.16s'
        }}
      />
      <div 
        style={{
          width: '6px',
          height: '6px',
          backgroundColor: 'var(--color-brand)',
          borderRadius: '50%',
          animation: 'bounce 1.4s infinite'
        }}
      />
    </Flex>
  </Flex>
); 
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

import { Flex } from "@kui/react";

interface CitationVisualContentProps {
  imageData: string;
  documentType: string;
}

export const CitationVisualContent = ({ 
  imageData, 
  documentType 
}: CitationVisualContentProps) => (
  <Flex justify="center">
    <img
      src={`data:image/png;base64,${imageData}`}
      alt={`Citation ${documentType}`}
      style={{
        maxWidth: '100%',
        height: 'auto',
        borderRadius: 'var(--border-radius-lg)',
        border: '1px solid var(--border-color-base)',
        boxShadow: 'var(--shadow-lg)'
      }}
    />
  </Flex>
);

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

import { useCitationUtils } from "../../hooks/useCitationUtils";
import { Flex, Text, Divider } from "@kui/react";
import { FileText, TrendingUp } from "lucide-react";

interface CitationMetadataProps {
  source?: string;
  score?: number;
}

export const CitationMetadata = ({ source, score }: CitationMetadataProps) => {
  const { formatScore } = useCitationUtils();

  if (!source && score === undefined) return null;

  return (
    <div style={{ paddingTop: 'var(--spacing-density-sm)' }}>
      <Divider />
      <Flex gap="density-md" style={{ paddingTop: 'var(--spacing-density-sm)', flexWrap: 'wrap' }}>
        {source && (
          <Flex align="center" gap="density-xs">
            <FileText size={12} style={{ color: 'var(--text-color-subtle)' }} />
            <Text kind="body/regular/xs" style={{ color: 'var(--text-color-subtle)' }}>
              Source: {source}
            </Text>
          </Flex>
        )}
        {score !== undefined && (
          <Flex align="center" gap="density-xs">
            <TrendingUp size={12} style={{ color: 'var(--text-color-subtle)' }} />
            <Text kind="body/regular/xs" style={{ color: 'var(--text-color-subtle)' }}>
              Relevance: {formatScore(score, 3)}
            </Text>
          </Flex>
        )}
      </Flex>
    </div>
  );
};

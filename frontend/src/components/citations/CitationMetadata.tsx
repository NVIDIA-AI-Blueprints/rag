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
import { FileText, TrendingUp, Workflow } from "lucide-react";

interface CitationMetadataProps {
  source?: string;
  score?: number;
  /**
   * Pipeline stage that produced this citation. Rendered as an opaque,
   * humanised string so future agentic stages display without code changes.
   */
  stage?: string;
}

export const CitationMetadata = ({ source, score, stage }: CitationMetadataProps) => {
  const { formatScore, formatStage } = useCitationUtils();

  if (!source && score === undefined && !stage) return null;

  return (
    <div style={{ paddingTop: 'var(--spacing-density-sm)' }}>
      <Divider />
      <Flex gap="density-md" style={{ paddingTop: 'var(--spacing-density-sm)', flexWrap: 'wrap' }}>
        {source && (
          <Flex align="center" gap="density-xs">
            <FileText size={14} style={{ color: 'var(--text-color-subtle)' }} />
            <Text kind="body/regular/sm" style={{ color: 'var(--text-color-subtle)' }}>
              Source: {source}
            </Text>
          </Flex>
        )}
        {score !== undefined && (
          <Flex align="center" gap="density-xs">
            <TrendingUp size={14} style={{ color: 'var(--text-color-subtle)' }} />
            <Text kind="body/regular/sm" style={{ color: 'var(--text-color-subtle)' }}>
              Relevance: {formatScore(score, 3)}
            </Text>
          </Flex>
        )}
        {stage && (
          <Flex align="center" gap="density-xs" data-testid="citation-stage-row" data-stage={stage}>
            <Workflow size={14} style={{ color: 'var(--text-color-subtle)' }} />
            <Text kind="body/regular/sm" style={{ color: 'var(--text-color-subtle)' }}>
              Pipeline stage: {formatStage(stage)}
            </Text>
          </Flex>
        )}
      </Flex>
    </div>
  );
};

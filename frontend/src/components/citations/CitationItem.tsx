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

import { useMemo, useCallback } from "react";
import type { Citation } from "../../types/chat";
import { useFileIcons } from "../../hooks/useFileIcons";
import { useCitationText } from "../../hooks/useCitationText";
import { useCitationExpansion } from "../../hooks/useCitationExpansion";
import { useCitationUtils } from "../../hooks/useCitationUtils";
import { CitationVisualContent } from "./CitationVisualContent";
import { CitationTextContent } from "./CitationTextContent";
import { CitationMetadata } from "./CitationMetadata";
import { ExpandChevron } from "../ui/ExpandChevron";
import { 
  Card, 
  Flex, 
  Stack, 
  Text, 
  Badge, 
  Block 
} from "@kui/react";

interface CitationItemProps {
  citation: Citation;
  index: number;
}

const CitationHeader = ({ 
  citation, 
  index, 
  isExpanded, 
  onClick 
}: {
  citation: Citation;
  index: number;
  isExpanded: boolean;
  onClick: () => void;
}) => {
  const { getFileIconByExtension } = useFileIcons();
  const { getAbridgedText } = useCitationText();
  const { formatScore } = useCitationUtils();

  // Keyboard accessibility handler
  const handleKeyDown = useCallback((event: React.KeyboardEvent<HTMLDivElement>) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      onClick();
    }
  }, [onClick]);

  return (
    <Block 
      padding="5"
      style={{ 
        cursor: 'pointer',
        userSelect: 'none'
      }}
      onClick={onClick}
      onKeyDown={handleKeyDown}
      role="button"
      tabIndex={0}
      aria-expanded={isExpanded}
      aria-label={`${isExpanded ? 'Collapse' : 'Expand'} citation from ${citation.source}`}
    >
      <Flex justify="between" align="center">
        <Flex align="start" gap="3">
          {/* Citation Number & Score Badges */}
          <Flex align="center" gap="2" style={{ flexShrink: 0 }}>
            <Badge 
              kind="solid" 
              color="green"
              data-testid="citation-badge"
            >
              #{index + 1}
            </Badge>
            {citation.score !== undefined && (
              <Badge 
                kind="outline" 
                color="green"
                data-testid="citation-score"
              >
                Score: {formatScore(citation.score, 2)}
              </Badge>
            )}
          </Flex>
          
          <Stack gap="1">
            <Flex align="center" gap="2">
              {getFileIconByExtension(citation.source, { size: 'sm' })}
              
              <Text 
                kind="body/bold/sm" 
                style={{ 
                  color: 'var(--text-color-inverse-brand)',
                  whiteSpace: 'nowrap',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis'
                }}
              >
                {citation.source}
              </Text>
              {citation.document_type && (
                <Badge kind="solid" color="gray">
                  {citation.document_type}
                </Badge>
              )}
            </Flex>
            {!isExpanded && (
              <Text 
                kind="body/regular/xs" 
                style={{ 
                  color: 'var(--text-color-subtle)',
                  marginLeft: '24px',
                  whiteSpace: 'nowrap',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  width: '50%'
                }}
              >
                {getAbridgedText(citation.text, 150)}
              </Text>
            )}
          </Stack>
        </Flex>
        <Flex align="center" gap="3" style={{ flexShrink: 0 }}>
          <ExpandChevron isExpanded={isExpanded} />
        </Flex>
      </Flex>
    </Block>
  );
};

const CitationContent = ({ citation }: { citation: Citation }) => {
  const { isVisualType } = useCitationUtils();
  const isVisual = isVisualType(citation.document_type);

  return (
    <Block
      style={{
        borderTop: '1px solid var(--border-color-base)',
        backgroundColor: 'var(--background-color-surface-sunken)'
      }}
    >
      <Block padding="5">
        <Stack gap="4">
          {isVisual ? (
            <CitationVisualContent 
              imageData={citation.text || ""} 
              documentType={citation.document_type} 
            />
          ) : (
            <CitationTextContent text={citation.text || ""} />
          )}
          
          <CitationMetadata 
            source={citation.source} 
          />
        </Stack>
      </Block>
    </Block>
  );
};

export default function CitationItem({ citation, index }: CitationItemProps) {
  const { generateCitationId } = useCitationUtils();
  
  const citationId = useMemo(() => 
    generateCitationId(citation, index), 
    [citation, index, generateCitationId]
  );
  
  const { isExpanded, toggle } = useCitationExpansion(citationId);

  return (
    <Card
      kind="float"
      interactive
      style={{
        borderRadius: '12px',
        backgroundColor: 'var(--background-color-surface-base)',
        overflow: 'hidden',
        transition: 'all 200ms ease'
      }}
    >
      <CitationHeader 
        citation={citation}
        index={index}
        isExpanded={isExpanded}
        onClick={toggle}
      />
      
      {isExpanded && <CitationContent citation={citation} />}
    </Card>
  );
}

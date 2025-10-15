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

interface CitationMetadataProps {
  source?: string;
  score?: number;
}

const DocumentIcon = () => (
  <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" 
          d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
  </svg>
);

const RelevanceIcon = () => (
  <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" 
          d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
  </svg>
);

export const CitationMetadata = ({ source, score }: CitationMetadataProps) => {
  const { formatScore } = useCitationUtils();

  // Don't render if no source and no score
  if (!source && score === undefined) return null;

  return (
    <div className="pt-3 border-t border-neutral-700/50">
      <div className="flex flex-wrap gap-3 text-xs text-gray-400">
        {source && (
          <div className="flex items-center gap-1">
            <DocumentIcon />
            <span>Source: {source}</span>
          </div>
        )}
        {score !== undefined && (
          <div className="flex items-center gap-1">
            <RelevanceIcon />
            <span>Relevance: {formatScore(score, 3)}</span>
          </div>
        )}
      </div>
    </div>
  );
}; 
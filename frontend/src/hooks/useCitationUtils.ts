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

import { useMemo } from "react";

const VISUAL_TYPES = ["image", "chart", "table"] as const;

export const useCitationUtils = () => {
  const isVisualType = useMemo(() => 
    (documentType: string) => VISUAL_TYPES.includes(documentType as typeof VISUAL_TYPES[number]),
    []
  );

  const formatScore = useMemo(() => 
    (score: number | string | undefined, precision = 2) => 
      score !== undefined ? Number(score).toFixed(precision) : 'N/A',
    []
  );

  const generateCitationId = useMemo(() => 
    (citation: { source?: string; text?: string }, index: number) =>
      `${citation.source || 'unknown'}-${index}-${citation.text?.slice(0, 20) || ''}`,
    []
  );

  return {
    isVisualType,
    formatScore,
    generateCitationId,
  };
}; 
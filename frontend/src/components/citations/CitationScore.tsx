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
import { Text, Stack } from "@kui/react";

interface CitationScoreProps {
  score: number | string | undefined;
  precision?: number;
}

export const CitationScore = ({ score, precision = 2 }: CitationScoreProps) => {
  const { formatScore } = useCitationUtils();

  if (score === undefined) return null;

  return (
    <Stack gap="density-xs" style={{ textAlign: 'right' }}>
      <Text kind="body/regular/xs" style={{ color: 'var(--text-color-subtle)' }}>Score</Text>
      <Text kind="label/bold/sm">{formatScore(score, precision)}</Text>
    </Stack>
  );
};

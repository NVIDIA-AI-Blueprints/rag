// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import { describe, it, expect } from 'vitest';
import { renderHook } from '@testing-library/react';
import { useCitationUtils } from '../useCitationUtils';

describe('useCitationUtils', () => {
  describe('formatStage', () => {
    it('returns empty string for undefined', () => {
      const { result } = renderHook(() => useCitationUtils());
      expect(result.current.formatStage(undefined)).toBe('');
    });

    it('returns empty string for an empty / whitespace-only stage', () => {
      const { result } = renderHook(() => useCitationUtils());
      expect(result.current.formatStage('')).toBe('');
      expect(result.current.formatStage('   ')).toBe('');
    });

    it('humanises snake_case stage identifiers', () => {
      const { result } = renderHook(() => useCitationUtils());
      expect(result.current.formatStage('initial_retrieval')).toBe('Initial retrieval');
      expect(result.current.formatStage('verify_execute')).toBe('Verify execute');
    });

    it('humanises kebab-case stage identifiers', () => {
      const { result } = renderHook(() => useCitationUtils());
      expect(result.current.formatStage('post-execute-verify')).toBe('Post execute verify');
    });

    it('preserves a single-word stage but capitalises it', () => {
      const { result } = renderHook(() => useCitationUtils());
      expect(result.current.formatStage('rag')).toBe('Rag');
      expect(result.current.formatStage('execute')).toBe('Execute');
    });

    it('handles future / unknown stage values without code changes', () => {
      const { result } = renderHook(() => useCitationUtils());
      // Whatever new stage the server adds, formatStage must not throw and
      // must return a non-empty humanised string.
      const future = 'plan_then_self_critique_v2';
      expect(result.current.formatStage(future)).toBe('Plan then self critique v2');
    });
  });

  describe('formatScore (regression)', () => {
    it('returns N/A for undefined and a fixed-precision string for numbers', () => {
      const { result } = renderHook(() => useCitationUtils());
      expect(result.current.formatScore(undefined)).toBe('N/A');
      expect(result.current.formatScore(0.876543)).toBe('0.88');
      expect(result.current.formatScore(0.876543, 3)).toBe('0.877');
    });
  });

  describe('isVisualType (regression)', () => {
    it('classifies image / chart / table as visual', () => {
      const { result } = renderHook(() => useCitationUtils());
      expect(result.current.isVisualType('image')).toBe(true);
      expect(result.current.isVisualType('chart')).toBe(true);
      expect(result.current.isVisualType('table')).toBe(true);
      expect(result.current.isVisualType('text')).toBe(false);
    });
  });
});

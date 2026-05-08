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
import { cleanRequestObject } from '../useMessageSubmit';
import type { GenerateRequest } from '../../types/requests';

describe('cleanRequestObject', () => {
  // Minimum payload to satisfy the GenerateRequest contract.
  const baseRequest: Partial<GenerateRequest> = {
    messages: [{ role: 'user', content: 'hello' }],
    use_knowledge_base: false,
  };

  it('omits undefined / null / empty values', () => {
    const cleaned = cleanRequestObject({
      ...baseRequest,
      temperature: undefined,
      filter_expr: '',
      stop: [],
      // null is allowed by the type for some fields and must be dropped
      agentic: null,
    });
    expect(cleaned).not.toHaveProperty('temperature');
    expect(cleaned).not.toHaveProperty('filter_expr');
    expect(cleaned).not.toHaveProperty('stop');
    expect(cleaned).not.toHaveProperty('agentic');
  });

  it('preserves explicit `agentic: true` (force agentic)', () => {
    const cleaned = cleanRequestObject({
      ...baseRequest,
      agentic: true,
    });
    expect(cleaned.agentic).toBe(true);
  });

  it('preserves explicit `agentic: false` (force standard) — must NOT be dropped', () => {
    // This is the safety-critical guarantee: if we dropped `false`, the
    // server's CONFIG.enable_agentic_rag would silently take over and the
    // user's "Standard" choice would be ignored.
    const cleaned = cleanRequestObject({
      ...baseRequest,
      agentic: false,
    });
    expect(cleaned).toHaveProperty('agentic', false);
  });

  it('omits `agentic` when undefined (defensive — FE always sends true/false now)', () => {
    // Since the "auto" mode was removed (see AgenticMode), the FE never
    // hands `cleanRequestObject` an `agentic: undefined`. This regression
    // guard keeps the cleaner honest if a caller ever passes one anyway.
    const cleaned = cleanRequestObject({
      ...baseRequest,
      agentic: undefined,
    });
    expect(cleaned).not.toHaveProperty('agentic');
  });

  it('preserves other always-include feature toggles set to false', () => {
    // Regression guard for the existing alwaysInclude list.
    const cleaned = cleanRequestObject({
      ...baseRequest,
      enable_reranker: false,
      enable_citations: false,
      enable_query_rewriting: false,
      enable_guardrails: false,
      enable_vlm_inference: false,
      enable_filter_generator: false,
    });
    expect(cleaned.enable_reranker).toBe(false);
    expect(cleaned.enable_citations).toBe(false);
    expect(cleaned.enable_query_rewriting).toBe(false);
    expect(cleaned.enable_guardrails).toBe(false);
    expect(cleaned.enable_vlm_inference).toBe(false);
    expect(cleaned.enable_filter_generator).toBe(false);
  });

  it('preserves always-include `use_knowledge_base: false`', () => {
    const cleaned = cleanRequestObject({
      ...baseRequest,
      use_knowledge_base: false,
    });
    expect(cleaned.use_knowledge_base).toBe(false);
  });

  it('preserves numeric and string values', () => {
    const cleaned = cleanRequestObject({
      ...baseRequest,
      temperature: 0.7,
      max_tokens: 1024,
      model: 'meta/llama-3.3-70b',
      collection_names: ['docs'],
    });
    expect(cleaned.temperature).toBe(0.7);
    expect(cleaned.max_tokens).toBe(1024);
    expect(cleaned.model).toBe('meta/llama-3.3-70b');
    expect(cleaned.collection_names).toEqual(['docs']);
  });

  it('drops booleans set to false that are NOT in the always-include list', () => {
    // `random_unknown_toggle` isn't a real GenerateRequest field — we cast
    // through unknown to verify the cleaner's behavior on arbitrary keys
    // without weakening GenerateRequest's type.
    const cleaned = cleanRequestObject({
      ...baseRequest,
      ...({ random_unknown_toggle: false } as unknown as Partial<GenerateRequest>),
    });
    expect(cleaned).not.toHaveProperty('random_unknown_toggle');
  });
});

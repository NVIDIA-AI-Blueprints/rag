// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, it, expect } from 'vitest';
import { renderHook } from '@testing-library/react';
import { useMarkdownRenderer } from '../useMarkdownRenderer';

describe('useMarkdownRenderer (XSS sanitization, CWE-79)', () => {
  it('strips <script> tags from rendered markdown', () => {
    const { result } = renderHook(() => useMarkdownRenderer());
    const html = result.current.renderMarkdown(
      'hello\n\n<script>window.__pwned=1</script>'
    );
    expect(html).not.toContain('<script');
    expect(html).not.toContain('window.__pwned');
  });

  it('removes inline event handler attributes', () => {
    const { result } = renderHook(() => useMarkdownRenderer());
    const html = result.current.renderMarkdown(
      '<img src=x onerror="window.__pwned=1">'
    );
    expect(html).not.toMatch(/onerror=/i);
  });

  it('neutralises javascript: URLs in links', () => {
    const { result } = renderHook(() => useMarkdownRenderer());
    const html = result.current.renderMarkdown(
      '[click](javascript:alert(1))'
    );
    expect(html.toLowerCase()).not.toContain('javascript:');
  });

  it('preserves benign markdown formatting', () => {
    const { result } = renderHook(() => useMarkdownRenderer());
    const html = result.current.renderMarkdown('**bold** and _italic_');
    expect(html).toContain('<strong>bold</strong>');
    expect(html).toContain('<em>italic</em>');
  });
});

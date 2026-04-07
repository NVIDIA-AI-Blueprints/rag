import { describe, it, expect } from 'vitest';
import { renderHook } from '../../test/utils';
import { useMarkdownRenderer } from '../useMarkdownRenderer';

describe('useMarkdownRenderer', () => {
  describe('XSS Prevention', () => {
    it('strips script tags from rendered output', () => {
      const { result } = renderHook(() => useMarkdownRenderer());
      const output = result.current.renderMarkdown(
        '<script>alert("xss")</script>'
      );
      expect(output).not.toContain('<script>');
      expect(output).not.toContain('alert(');
    });

    it('strips onerror event handlers from img tags', () => {
      const { result } = renderHook(() => useMarkdownRenderer());
      const output = result.current.renderMarkdown(
        '<img src=x onerror=alert(1)>'
      );
      expect(output).not.toContain('onerror');
    });

    it('strips javascript: URIs from links', () => {
      const { result } = renderHook(() => useMarkdownRenderer());
      const output = result.current.renderMarkdown(
        '<a href="javascript:alert(1)">click</a>'
      );
      expect(output).not.toContain('javascript:');
    });

    it('strips onload event handlers from SVG', () => {
      const { result } = renderHook(() => useMarkdownRenderer());
      const output = result.current.renderMarkdown(
        '<svg onload=alert(1)>'
      );
      expect(output).not.toContain('onload');
    });

    it('strips iframe tags', () => {
      const { result } = renderHook(() => useMarkdownRenderer());
      const output = result.current.renderMarkdown(
        '<iframe src="https://evil.com"></iframe>'
      );
      expect(output).not.toContain('<iframe');
    });
  });

  describe('Safe Content Preservation', () => {
    it('renders bold text correctly', () => {
      const { result } = renderHook(() => useMarkdownRenderer());
      const output = result.current.renderMarkdown('**bold**');
      expect(output).toContain('<strong>bold</strong>');
    });

    it('renders italic text correctly', () => {
      const { result } = renderHook(() => useMarkdownRenderer());
      const output = result.current.renderMarkdown('*italic*');
      expect(output).toContain('<em>italic</em>');
    });

    it('renders links correctly', () => {
      const { result } = renderHook(() => useMarkdownRenderer());
      const output = result.current.renderMarkdown('[link](https://example.com)');
      expect(output).toContain('href="https://example.com"');
      expect(output).toContain('link');
    });

    it('renders lists correctly', () => {
      const { result } = renderHook(() => useMarkdownRenderer());
      const output = result.current.renderMarkdown('- item 1\n- item 2');
      expect(output).toContain('<li>');
      expect(output).toContain('item 1');
    });
  });
});

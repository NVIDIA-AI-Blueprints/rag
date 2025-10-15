import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render } from '../../../test/utils';
import { MessageContent } from '../MessageContent';

// Mock the markdown renderer hook
const mockRenderMarkdown = vi.fn();
vi.mock('../../../hooks/useMarkdownRenderer', () => ({
  useMarkdownRenderer: () => ({
    renderMarkdown: mockRenderMarkdown
  })
}));

describe('MessageContent', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockRenderMarkdown.mockReturnValue('<p>rendered content</p>');
  });

  describe('Content Processing', () => {
    it('calls renderMarkdown with provided content', () => {
      render(<MessageContent content="original text" />);
      
      expect(mockRenderMarkdown).toHaveBeenCalledWith('original text');
    });

    it('handles empty content', () => {
      render(<MessageContent content="" />);
      
      expect(mockRenderMarkdown).toHaveBeenCalledWith('');
    });

    it('handles different content types', () => {
      const content = '# Header\n\nSome **bold** text';
      render(<MessageContent content={content} />);
      
      expect(mockRenderMarkdown).toHaveBeenCalledWith(content);
    });

    it('processes content once per render', () => {
      render(<MessageContent content="test content" />);
      
      expect(mockRenderMarkdown).toHaveBeenCalledTimes(1);
    });
  });

  describe('HTML Rendering', () => {
    it('renders processed HTML content', () => {
      mockRenderMarkdown.mockReturnValue('<strong>bold text</strong>');
      
      const { container } = render(<MessageContent content="test" />);
      
      expect(container.innerHTML).toContain('<strong>bold text</strong>');
    });

    it('uses dangerouslySetInnerHTML for content', () => {
      mockRenderMarkdown.mockReturnValue('<em>italic</em>');
      
      const { container } = render(<MessageContent content="test" />);
      
      expect(container.innerHTML).toContain('<em>italic</em>');
    });

    it('handles complex HTML structures', () => {
      mockRenderMarkdown.mockReturnValue('<div><p>Paragraph</p><ul><li>Item</li></ul></div>');
      
      const { container } = render(<MessageContent content="test" />);
      
      expect(container.innerHTML).toContain('<div><p>Paragraph</p><ul><li>Item</li></ul></div>');
    });
  });

  describe('CSS Classes', () => {
    it('uses default CSS classes when none provided', () => {
      const { container } = render(<MessageContent content="test" />);
      
      const contentDiv = container.firstElementChild;
      expect(contentDiv).toHaveClass('prose', 'prose-invert', 'max-w-none', 'text-sm');
    });

    it('uses custom CSS classes when provided', () => {
      const { container } = render(<MessageContent content="test" className="custom-class" />);
      
      const contentDiv = container.firstElementChild;
      expect(contentDiv).toHaveClass('custom-class');
      expect(contentDiv).not.toHaveClass('prose');
    });

    it('applies different custom classes', () => {
      const { container } = render(<MessageContent content="test" className="large-text bold" />);
      
      const contentDiv = container.firstElementChild;
      expect(contentDiv).toHaveClass('large-text', 'bold');
    });
  });

  describe('Content Updates', () => {
    it('processes new content when changed', () => {
      const { rerender } = render(<MessageContent content="initial" />);
      
      rerender(<MessageContent content="updated" />);
      
      expect(mockRenderMarkdown).toHaveBeenCalledWith('initial');
      expect(mockRenderMarkdown).toHaveBeenCalledWith('updated');
      expect(mockRenderMarkdown).toHaveBeenCalledTimes(2);
    });

    it('updates rendered output when content changes', () => {
      mockRenderMarkdown.mockReturnValue('<p>first</p>');
      const { container, rerender } = render(<MessageContent content="first" />);
      
      mockRenderMarkdown.mockReturnValue('<p>second</p>');
      rerender(<MessageContent content="second" />);
      
      expect(container.innerHTML).toContain('<p>second</p>');
    });
  });
}); 
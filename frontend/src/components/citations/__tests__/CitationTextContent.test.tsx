import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render } from '../../../test/utils';
import { CitationTextContent } from '../CitationTextContent';

// Mock hooks
const mockRenderMarkdown = vi.fn();
const mockToMarkdown = vi.fn();

vi.mock('../../../hooks/useMarkdownRenderer', () => ({
  useMarkdownRenderer: () => ({
    renderMarkdown: mockRenderMarkdown
  })
}));

vi.mock('../../../hooks/useCitationText', () => ({
  useCitationText: () => ({
    toMarkdown: mockToMarkdown
  })
}));

describe('CitationTextContent', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockToMarkdown.mockReturnValue('markdown text');
    mockRenderMarkdown.mockReturnValue('<p>rendered html</p>');
  });

  describe('Text Processing', () => {
    it('calls toMarkdown with provided text', () => {
      render(<CitationTextContent text="original text" />);
      
      expect(mockToMarkdown).toHaveBeenCalledWith('original text');
    });

    it('calls renderMarkdown with markdown output', () => {
      mockToMarkdown.mockReturnValue('converted markdown');
      
      render(<CitationTextContent text="test" />);
      
      expect(mockRenderMarkdown).toHaveBeenCalledWith('converted markdown');
    });

    it('handles empty text', () => {
      render(<CitationTextContent text="" />);
      
      expect(mockToMarkdown).toHaveBeenCalledWith('');
    });

    it('handles different text inputs', () => {
      render(<CitationTextContent text="different content" />);
      
      expect(mockToMarkdown).toHaveBeenCalledWith('different content');
    });
  });

  describe('HTML Rendering', () => {
    it('renders processed HTML content', () => {
      mockRenderMarkdown.mockReturnValue('<strong>bold text</strong>');
      
      const { container } = render(<CitationTextContent text="test" />);
      
      expect(container.innerHTML).toContain('<strong>bold text</strong>');
    });

    it('renders different HTML outputs', () => {
      mockRenderMarkdown.mockReturnValue('<em>italic text</em>');
      
      const { container } = render(<CitationTextContent text="test" />);
      
      expect(container.innerHTML).toContain('<em>italic text</em>');
    });

    it('uses dangerouslySetInnerHTML for content', () => {
      mockRenderMarkdown.mockReturnValue('<div>content</div>');
      
      const { container } = render(<CitationTextContent text="test" />);
      
      const contentDiv = container.querySelector('div[class*="text-gray-300"]');
      expect(contentDiv).toBeInTheDocument();
    });
  });

  describe('Processing Chain', () => {
    it('processes text through both hooks in sequence', () => {
      mockToMarkdown.mockReturnValue('step1 output');
      mockRenderMarkdown.mockReturnValue('step2 output');
      
      const { container } = render(<CitationTextContent text="input text" />);
      
      expect(mockToMarkdown).toHaveBeenCalledWith('input text');
      expect(mockRenderMarkdown).toHaveBeenCalledWith('step1 output');
      expect(container.innerHTML).toContain('step2 output');
    });

    it('maintains processing order with different inputs', () => {
      mockToMarkdown.mockReturnValue('markdown result');
      mockRenderMarkdown.mockReturnValue('html result');
      
      render(<CitationTextContent text="source text" />);
      
      expect(mockToMarkdown).toHaveBeenCalledBefore(mockRenderMarkdown);
    });
  });
}); 
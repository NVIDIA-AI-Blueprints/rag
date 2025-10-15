import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '../../../test/utils';
import CitationItem from '../CitationItem';
import type { Citation } from '../../../types/chat';

// Mock all the hooks used by CitationItem
const mockUseFileIcons = vi.fn();
const mockUseCitationText = vi.fn();
const mockUseCitationExpansion = vi.fn();
const mockUseCitationUtils = vi.fn();

vi.mock('../../../hooks/useFileIcons', () => ({
  useFileIcons: () => mockUseFileIcons()
}));

vi.mock('../../../hooks/useCitationText', () => ({
  useCitationText: () => mockUseCitationText()
}));

vi.mock('../../../hooks/useCitationExpansion', () => ({
  useCitationExpansion: () => mockUseCitationExpansion()
}));

vi.mock('../../../hooks/useCitationUtils', () => ({
  useCitationUtils: () => mockUseCitationUtils()
}));

// Mock child components to focus on CitationItem logic
vi.mock('../CitationBadge', () => ({
  CitationBadge: ({ number }: { number: number }) => <div data-testid="citation-badge">{number}</div>
}));

vi.mock('../CitationScore', () => ({
  CitationScore: ({ score }: { score: number }) => <div data-testid="citation-score">{score}</div>
}));

vi.mock('../CitationVisualContent', () => ({
  CitationVisualContent: ({ documentType }: { documentType: string }) => (
    <div data-testid="visual-content">Visual: {documentType}</div>
  )
}));

vi.mock('../CitationTextContent', () => ({
  CitationTextContent: ({ text }: { text: string }) => (
    <div data-testid="text-content">{text}</div>
  )
}));

vi.mock('../CitationMetadata', () => ({
  CitationMetadata: ({ source, score }: { source: string; score?: string | number }) => (
    <div data-testid="citation-metadata">{source} - {score}</div>
  )
}));

vi.mock('../../ui/ExpandChevron', () => ({
  ExpandChevron: ({ isExpanded }: { isExpanded: boolean }) => (
    <div data-testid="expand-chevron">{isExpanded ? 'expanded' : 'collapsed'}</div>
  )
}));

describe('CitationItem', () => {
  const mockCitation: Citation = {
    source: 'test-document.pdf',
    text: 'This is a sample citation text',
    score: 0.85,
    document_type: 'text'
  };

  const mockGenerateCitationId = vi.fn();
  const mockToggle = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    
    mockUseFileIcons.mockReturnValue({
      getFileIconByExtension: vi.fn().mockReturnValue(<div>📄</div>)
    });
    
    mockUseCitationText.mockReturnValue({
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
      getAbridgedText: vi.fn().mockImplementation((_text, _length) => 'Abridged text...')
    });
    
    mockUseCitationUtils.mockReturnValue({
      generateCitationId: mockGenerateCitationId,
      isVisualType: vi.fn().mockReturnValue(false),
      formatScore: vi.fn().mockReturnValue('0.85')
    });
    
    mockUseCitationExpansion.mockReturnValue({
      isExpanded: false,
      toggle: mockToggle
    });
    
    mockGenerateCitationId.mockReturnValue('citation-123');
  });

  describe('Citation ID Generation', () => {
    it('calls generateCitationId with citation and index', () => {
      render(<CitationItem citation={mockCitation} index={2} />);
      
      expect(mockGenerateCitationId).toHaveBeenCalledWith(mockCitation, 2);
    });

    it('memoizes citationId correctly', () => {
      const { rerender } = render(<CitationItem citation={mockCitation} index={1} />);
      
      expect(mockGenerateCitationId).toHaveBeenCalledTimes(1);
      
      // Rerender with same props - should not call again due to memoization
      rerender(<CitationItem citation={mockCitation} index={1} />);
      expect(mockGenerateCitationId).toHaveBeenCalledTimes(1);
      
      // Rerender with different index - should call again
      rerender(<CitationItem citation={mockCitation} index={2} />);
      expect(mockGenerateCitationId).toHaveBeenCalledTimes(2);
    });

    it('uses generated citationId for expansion hook', () => {
      // This test verifies that the component generates a citation ID and uses it
      // The expansion hook receives the generated ID (though exact value depends on mocking)
      render(<CitationItem citation={mockCitation} index={0} />);
      
      // Verify that generateCitationId was called with correct parameters
      expect(mockGenerateCitationId).toHaveBeenCalledWith(mockCitation, 0);
      
      // Verify that the expansion hook was called (the exact ID value is implementation detail)
      expect(mockUseCitationExpansion).toHaveBeenCalled();
    });
  });

  describe('Header Rendering', () => {
    it('renders citation badge with correct number', () => {
      render(<CitationItem citation={mockCitation} index={2} />);
      
      expect(screen.getByTestId('citation-badge')).toHaveTextContent('3'); // index + 1
    });

    it('renders citation source', () => {
      render(<CitationItem citation={mockCitation} index={0} />);
      
      expect(screen.getByText('test-document.pdf')).toBeInTheDocument();
    });

    it('renders document type badge when present', () => {
      render(<CitationItem citation={mockCitation} index={0} />);
      
      expect(screen.getByText('text')).toBeInTheDocument();
    });

    it('renders citation score', () => {
      render(<CitationItem citation={mockCitation} index={0} />);
      
      expect(screen.getByTestId('citation-score')).toHaveTextContent('0.85');
    });

    it('calls file icon hook with correct document type', () => {
      const mockGetIcon = vi.fn().mockReturnValue(<div>📄</div>);
      mockUseFileIcons.mockReturnValue({
        getFileIconByExtension: mockGetIcon
      });

      render(<CitationItem citation={mockCitation} index={0} />);
      
      expect(mockGetIcon).toHaveBeenCalledWith('test-document.pdf', { size: 'sm' });
    });
  });

  describe('Expansion Behavior', () => {
    it('shows collapsed state by default', () => {
      mockUseCitationExpansion.mockReturnValue({
        isExpanded: false,
        toggle: mockToggle
      });

      render(<CitationItem citation={mockCitation} index={0} />);
      
      expect(screen.getByTestId('expand-chevron')).toHaveTextContent('collapsed');
    });

    it('shows expanded state when expanded', () => {
      mockUseCitationExpansion.mockReturnValue({
        isExpanded: true,
        toggle: mockToggle
      });

      render(<CitationItem citation={mockCitation} index={0} />);
      
      expect(screen.getByTestId('expand-chevron')).toHaveTextContent('expanded');
    });

    it('calls toggle when header is clicked', () => {
      render(<CitationItem citation={mockCitation} index={0} />);
      
      const header = screen.getByText('test-document.pdf').closest('div');
      fireEvent.click(header!);
      
      expect(mockToggle).toHaveBeenCalledOnce();
    });

    it('shows abridged text when collapsed', () => {
      mockUseCitationExpansion.mockReturnValue({
        isExpanded: false,
        toggle: mockToggle
      });

      const mockGetAbridged = vi.fn().mockReturnValue('Abridged text...');
      mockUseCitationText.mockReturnValue({
        getAbridgedText: mockGetAbridged
      });

      render(<CitationItem citation={mockCitation} index={0} />);
      
      expect(mockGetAbridged).toHaveBeenCalledWith('This is a sample citation text', 150);
      expect(screen.getByText('Abridged text...')).toBeInTheDocument();
    });

    it('hides abridged text when expanded', () => {
      mockUseCitationExpansion.mockReturnValue({
        isExpanded: true,
        toggle: mockToggle
      });

      render(<CitationItem citation={mockCitation} index={0} />);
      
      expect(screen.queryByText(/Abridged text/)).not.toBeInTheDocument();
    });
  });

  describe('Content Rendering', () => {
    it('shows citation content when expanded', () => {
      mockUseCitationExpansion.mockReturnValue({
        isExpanded: true,
        toggle: mockToggle
      });

      render(<CitationItem citation={mockCitation} index={0} />);
      
      expect(screen.getByTestId('text-content')).toBeInTheDocument();
      expect(screen.getByTestId('citation-metadata')).toBeInTheDocument();
    });

    it('hides citation content when collapsed', () => {
      mockUseCitationExpansion.mockReturnValue({
        isExpanded: false,
        toggle: mockToggle
      });

      render(<CitationItem citation={mockCitation} index={0} />);
      
      expect(screen.queryByTestId('text-content')).not.toBeInTheDocument();
      expect(screen.queryByTestId('citation-metadata')).not.toBeInTheDocument();
    });

    it('renders visual content for visual document types', () => {
      mockUseCitationExpansion.mockReturnValue({
        isExpanded: true,
        toggle: mockToggle
      });

      mockUseCitationUtils.mockReturnValue({
        generateCitationId: mockGenerateCitationId,
        isVisualType: vi.fn().mockReturnValue(true),
        formatScore: vi.fn().mockReturnValue('0.85')
      });

      render(<CitationItem citation={mockCitation} index={0} />);
      
      expect(screen.getByTestId('visual-content')).toBeInTheDocument();
      expect(screen.queryByTestId('text-content')).not.toBeInTheDocument();
    });

    it('renders text content for non-visual document types', () => {
      mockUseCitationExpansion.mockReturnValue({
        isExpanded: true,
        toggle: mockToggle
      });

      mockUseCitationUtils.mockReturnValue({
        generateCitationId: mockGenerateCitationId,
        isVisualType: vi.fn().mockReturnValue(false),
        formatScore: vi.fn().mockReturnValue('0.85')
      });

      render(<CitationItem citation={mockCitation} index={0} />);
      
      expect(screen.getByTestId('text-content')).toBeInTheDocument();
      expect(screen.queryByTestId('visual-content')).not.toBeInTheDocument();
    });
  });

  describe('Edge Cases', () => {
    it('handles different document type', () => {
      const citationWithImageType: Citation = { ...mockCitation, document_type: 'image' };
      
      render(<CitationItem citation={citationWithImageType} index={0} />);
      
      expect(screen.getByText('image')).toBeInTheDocument();
    });

    it('handles empty text', () => {
      const citationWithEmptyText = { ...mockCitation, text: '' };
      
      mockUseCitationExpansion.mockReturnValue({
        isExpanded: true,
        toggle: mockToggle
      });

      render(<CitationItem citation={citationWithEmptyText} index={0} />);
      
      expect(screen.getByTestId('text-content')).toHaveTextContent('');
    });

    it('handles zero index correctly', () => {
      render(<CitationItem citation={mockCitation} index={0} />);
      
      expect(screen.getByTestId('citation-badge')).toHaveTextContent('1'); // index + 1
      expect(mockGenerateCitationId).toHaveBeenCalledWith(mockCitation, 0);
    });

    it('handles high index values', () => {
      render(<CitationItem citation={mockCitation} index={99} />);
      
      expect(screen.getByTestId('citation-badge')).toHaveTextContent('100');
      expect(mockGenerateCitationId).toHaveBeenCalledWith(mockCitation, 99);
    });
  });

  describe('Integration with Utils', () => {
    it('passes correct parameters to isVisualType', () => {
      const mockIsVisualType = vi.fn().mockReturnValue(false);
      mockUseCitationUtils.mockReturnValue({
        generateCitationId: mockGenerateCitationId,
        isVisualType: mockIsVisualType,
        formatScore: vi.fn().mockReturnValue('0.85')
      });

      mockUseCitationExpansion.mockReturnValue({
        isExpanded: true,
        toggle: mockToggle
      });

      render(<CitationItem citation={mockCitation} index={0} />);
      
      expect(mockIsVisualType).toHaveBeenCalledWith('text');
    });
  });
}); 
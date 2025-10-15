import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '../../../test/utils';
import { CitationButton } from '../CitationButton';

// Mock hooks
const mockShowCitations = vi.fn();
const mockFormatCitationCount = vi.fn();

vi.mock('../../../hooks/useCitations', () => ({
  useCitations: () => ({
    showCitations: mockShowCitations,
    formatCitationCount: mockFormatCitationCount
  })
}));

describe('CitationButton', () => {
  const mockCitations = [
    { source: 'doc1.pdf', text: 'content 1', score: 0.9, document_type: 'text' as const },
    { source: 'doc2.pdf', text: 'content 2', score: 0.8, document_type: 'text' as const }
  ];

  beforeEach(() => {
    vi.clearAllMocks();
    mockFormatCitationCount.mockReturnValue('2 sources');
  });

  describe('Click Behavior', () => {
    it('calls showCitations when button clicked', () => {
      render(<CitationButton citations={mockCitations} />);
      
      fireEvent.click(screen.getByRole('button'));
      
      expect(mockShowCitations).toHaveBeenCalledWith(mockCitations);
    });

    it('passes correct citations to showCitations', () => {
      const differentCitations = [{ source: 'doc3.pdf', text: 'content 3', score: 0.7, document_type: 'text' as const }];
      
      render(<CitationButton citations={differentCitations} />);
      
      fireEvent.click(screen.getByRole('button'));
      
      expect(mockShowCitations).toHaveBeenCalledWith(differentCitations);
    });

    it('works with empty citations array', () => {
      render(<CitationButton citations={[]} />);
      
      fireEvent.click(screen.getByRole('button'));
      
      expect(mockShowCitations).toHaveBeenCalledWith([]);
    });

    it('can be clicked multiple times', () => {
      render(<CitationButton citations={mockCitations} />);
      
      const button = screen.getByRole('button');
      fireEvent.click(button);
      fireEvent.click(button);
      
      expect(mockShowCitations).toHaveBeenCalledTimes(2);
    });
  });

  describe('Citation Count Display', () => {
    it('calls formatCitationCount with correct length', () => {
      render(<CitationButton citations={mockCitations} />);
      
      expect(mockFormatCitationCount).toHaveBeenCalledWith(2);
    });

    it('displays formatted citation count', () => {
      mockFormatCitationCount.mockReturnValue('3 sources');
      
      render(<CitationButton citations={[{ text: '', source: '', document_type: 'text' }, { text: '', source: '', document_type: 'text' }, { text: '', source: '', document_type: 'text' }]} />);
      
      expect(screen.getByText('3 sources')).toBeInTheDocument();
    });

    it('handles zero citations', () => {
      mockFormatCitationCount.mockReturnValue('No sources');
      
      render(<CitationButton citations={[]} />);
      
      expect(mockFormatCitationCount).toHaveBeenCalledWith(0);
      expect(screen.getByText('No sources')).toBeInTheDocument();
    });

    it('handles single citation', () => {
      mockFormatCitationCount.mockReturnValue('1 source');
      
      render(<CitationButton citations={[mockCitations[0]]} />);
      
      expect(mockFormatCitationCount).toHaveBeenCalledWith(1);
      expect(screen.getByText('1 source')).toBeInTheDocument();
    });
  });

  describe('Button Rendering', () => {
    it('renders as clickable button', () => {
      render(<CitationButton citations={mockCitations} />);
      
      expect(screen.getByRole('button')).toBeInTheDocument();
    });

    it('contains formatted count text', () => {
      mockFormatCitationCount.mockReturnValue('View 5 citations');
      
      render(<CitationButton citations={[{ text: '', source: '', document_type: 'text' }, { text: '', source: '', document_type: 'text' }, { text: '', source: '', document_type: 'text' }, { text: '', source: '', document_type: 'text' }, { text: '', source: '', document_type: 'text' }]} />);
      
      expect(screen.getByText('View 5 citations')).toBeInTheDocument();
    });
  });
}); 
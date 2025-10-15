import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '../../../test/utils';
import { CitationMetadata } from '../CitationMetadata';

// Mock the citation utils hook
const mockFormatScore = vi.fn();
vi.mock('../../../hooks/useCitationUtils', () => ({
  useCitationUtils: () => ({
    formatScore: mockFormatScore
  })
}));

describe('CitationMetadata', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockFormatScore.mockReturnValue('0.850');
  });

  describe('Conditional Rendering', () => {
    it('renders nothing when no source and no score', () => {
      const { container } = render(<CitationMetadata />);
      
      expect(container.firstChild).toBeNull();
    });

    it('renders nothing when source is undefined and score is undefined', () => {
      const { container } = render(<CitationMetadata source={undefined} score={undefined} />);
      
      expect(container.firstChild).toBeNull();
    });

    it('renders when source is provided', () => {
      render(<CitationMetadata source="document.pdf" />);
      
      expect(screen.getByText('Source: document.pdf')).toBeInTheDocument();
    });

    it('renders when score is provided', () => {
      mockFormatScore.mockReturnValue('0.95');
      
      render(<CitationMetadata score={0.95} />);
      
      expect(screen.getByText('Relevance: 0.95')).toBeInTheDocument();
    });

    it('renders both source and score when both provided', () => {
      mockFormatScore.mockReturnValue('0.80');
      
      render(<CitationMetadata source="test.pdf" score={0.8} />);
      
      expect(screen.getByText('Source: test.pdf')).toBeInTheDocument();
      expect(screen.getByText('Relevance: 0.80')).toBeInTheDocument();
    });
  });

  describe('Source Display', () => {
    it('displays correct source name', () => {
      render(<CitationMetadata source="document.pdf" />);
      
      expect(screen.getByText('Source: document.pdf')).toBeInTheDocument();
    });

    it('displays different source names', () => {
      render(<CitationMetadata source="another-file.txt" />);
      
      expect(screen.getByText('Source: another-file.txt')).toBeInTheDocument();
    });

    it('does not display source when not provided', () => {
      render(<CitationMetadata score={0.5} />);
      
      expect(screen.queryByText(/Source:/)).not.toBeInTheDocument();
    });
  });

  describe('Score Display', () => {
    it('calls formatScore with correct parameters', () => {
      render(<CitationMetadata score={0.123456} />);
      
      expect(mockFormatScore).toHaveBeenCalledWith(0.123456, 3);
    });

    it('displays formatted score with Relevance label', () => {
      mockFormatScore.mockReturnValue('0.789');
      
      render(<CitationMetadata score={0.789} />);
      
      expect(screen.getByText('Relevance: 0.789')).toBeInTheDocument();
    });

    it('handles string scores', () => {
      render(<CitationMetadata score={3} />);
      
      expect(mockFormatScore).toHaveBeenCalledWith(3, 3);
    });

    it('handles zero score', () => {
      mockFormatScore.mockReturnValue('0.000');
      
      render(<CitationMetadata score={0} />);
      
      expect(screen.getByText('Relevance: 0.000')).toBeInTheDocument();
    });

    it('does not display score when not provided', () => {
      render(<CitationMetadata source="test.pdf" />);
      
      expect(screen.queryByText(/Relevance:/)).not.toBeInTheDocument();
    });
  });

  describe('Combined Display', () => {
    it('shows both elements in correct format', () => {
      mockFormatScore.mockReturnValue('0.92');
      
      render(<CitationMetadata source="combined.pdf" score={0.92} />);
      
      expect(screen.getByText('Source: combined.pdf')).toBeInTheDocument();
      expect(screen.getByText('Relevance: 0.92')).toBeInTheDocument();
    });

    it('formats precision correctly for scores', () => {
      render(<CitationMetadata source="test.pdf" score={0.123456789} />);
      
      expect(mockFormatScore).toHaveBeenCalledWith(0.123456789, 3);
    });
  });
}); 
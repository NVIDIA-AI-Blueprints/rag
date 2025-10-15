import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '../../../test/utils';
import { CitationScore } from '../CitationScore';

// Mock the citation utils hook
const mockFormatScore = vi.fn();
vi.mock('../../../hooks/useCitationUtils', () => ({
  useCitationUtils: () => ({
    formatScore: mockFormatScore
  })
}));

describe('CitationScore', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockFormatScore.mockReturnValue('0.85');
  });

  describe('Conditional Rendering', () => {
    it('renders nothing when score is undefined', () => {
      const { container } = render(<CitationScore score={undefined} />);
      
      expect(container.firstChild).toBeNull();
    });

    it('renders score when score is provided', () => {
      mockFormatScore.mockReturnValue('0.95');
      
      render(<CitationScore score={0.95} />);
      
      expect(screen.getByText('Score')).toBeInTheDocument();
      expect(screen.getByText('0.95')).toBeInTheDocument();
    });

    it('renders score when score is zero', () => {
      mockFormatScore.mockReturnValue('0.00');
      
      render(<CitationScore score={0} />);
      
      expect(screen.getByText('Score')).toBeInTheDocument();
      expect(screen.getByText('0.00')).toBeInTheDocument();
    });

    it('renders score when score is string', () => {
      mockFormatScore.mockReturnValue('high');
      
      render(<CitationScore score="high" />);
      
      expect(screen.getByText('Score')).toBeInTheDocument();
      expect(screen.getByText('high')).toBeInTheDocument();
    });
  });

  describe('Score Formatting', () => {
    it('calls formatScore with correct parameters', () => {
      render(<CitationScore score={0.123456} precision={2} />);
      
      expect(mockFormatScore).toHaveBeenCalledWith(0.123456, 2);
    });

    it('uses default precision when not provided', () => {
      render(<CitationScore score={0.5} />);
      
      expect(mockFormatScore).toHaveBeenCalledWith(0.5, 2);
    });

    it('uses custom precision when provided', () => {
      render(<CitationScore score={0.123456} precision={4} />);
      
      expect(mockFormatScore).toHaveBeenCalledWith(0.123456, 4);
    });

    it('handles string scores', () => {
      render(<CitationScore score="medium" precision={2} />);
      
      expect(mockFormatScore).toHaveBeenCalledWith('medium', 2);
    });
  });

  describe('Content Display', () => {
    it('displays Score label', () => {
      render(<CitationScore score={0.75} />);
      
      expect(screen.getByText('Score')).toBeInTheDocument();
    });

    it('displays formatted score value', () => {
      mockFormatScore.mockReturnValue('0.89');
      
      render(<CitationScore score={0.89} />);
      
      expect(screen.getByText('0.89')).toBeInTheDocument();
    });

    it('displays different formatted values', () => {
      mockFormatScore.mockReturnValue('1.00');
      
      render(<CitationScore score={1} />);
      
      expect(screen.getByText('1.00')).toBeInTheDocument();
    });
  });
}); 
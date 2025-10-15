import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '../../../test/utils';
import Citations from '../Citations';
import type { Citation } from '../../../types/chat';

// Mock CitationItem component
vi.mock('../../citations/CitationItem', () => ({
  default: ({ citation, index }: { citation: Citation; index: number }) => (
    <div data-testid="citation-item">
      Citation {index + 1}: {citation.source}
    </div>
  )
}));

describe('Citations', () => {
  describe('Empty State', () => {
    it('shows empty state when no citations provided', () => {
      render(<Citations />);
      
      expect(screen.getByText('No Citations Available')).toBeInTheDocument();
      expect(screen.getByText(/No sources were found for this response/)).toBeInTheDocument();
    });

    it('shows empty state when citations array is empty', () => {
      render(<Citations citations={[]} />);
      
      expect(screen.getByText('No Citations Available')).toBeInTheDocument();
    });

    it('does not show citation items when empty', () => {
      render(<Citations citations={[]} />);
      
      expect(screen.queryByTestId('citation-item')).not.toBeInTheDocument();
    });
  });

  describe('Citations Display', () => {
    const mockCitations = [
      { source: 'doc1.pdf', text: 'content 1', score: 0.9, document_type: 'text' as const },
      { source: 'doc2.pdf', text: 'content 2', score: 0.8, document_type: 'text' as const }
    ];

    it('renders citation items when citations provided', () => {
      render(<Citations citations={mockCitations} />);
      
      expect(screen.getByText('Citation 1: doc1.pdf')).toBeInTheDocument();
      expect(screen.getByText('Citation 2: doc2.pdf')).toBeInTheDocument();
    });

    it('renders correct number of citation items', () => {
      render(<Citations citations={mockCitations} />);
      
      const items = screen.getAllByTestId('citation-item');
      expect(items).toHaveLength(2);
    });

    it('does not show empty state when citations exist', () => {
      render(<Citations citations={mockCitations} />);
      
      expect(screen.queryByText('No Citations Available')).not.toBeInTheDocument();
    });

    it('passes correct props to CitationItem', () => {
      render(<Citations citations={mockCitations} />);
      
      expect(screen.getByText('Citation 1: doc1.pdf')).toBeInTheDocument();
      expect(screen.getByText('Citation 2: doc2.pdf')).toBeInTheDocument();
    });

    it('handles single citation', () => {
      const singleCitation = [mockCitations[0]];
      
      render(<Citations citations={singleCitation} />);
      
      expect(screen.getByText('Citation 1: doc1.pdf')).toBeInTheDocument();
      expect(screen.getAllByTestId('citation-item')).toHaveLength(1);
    });

    it('handles multiple citations', () => {
      const multipleCitations = [
        ...mockCitations,
        { source: 'doc3.pdf', text: 'content 3', score: 0.7, document_type: 'text' as const }
      ];
      
      render(<Citations citations={multipleCitations} />);
      
      expect(screen.getAllByTestId('citation-item')).toHaveLength(3);
      expect(screen.getByText('Citation 3: doc3.pdf')).toBeInTheDocument();
    });
  });

  describe('Conditional Rendering', () => {
    it('switches from empty state to citations when citations added', () => {
      const mockCitations = [
        { source: 'doc1.pdf', text: 'content', score: 0.9, document_type: 'text' as const }
      ];

      const { rerender } = render(<Citations citations={[]} />);
      expect(screen.getByText('No Citations Available')).toBeInTheDocument();

      rerender(<Citations citations={mockCitations} />);
      expect(screen.queryByText('No Citations Available')).not.toBeInTheDocument();
      expect(screen.getByTestId('citation-item')).toBeInTheDocument();
    });

    it('shows correct content based on citations array length', () => {
      const mockCitations = [
        { source: 'doc1.pdf', text: 'content', score: 0.9, document_type: 'text' as const }
      ];

      const { rerender } = render(<Citations citations={mockCitations} />);
      expect(screen.getByTestId('citation-item')).toBeInTheDocument();

      rerender(<Citations citations={[]} />);
      expect(screen.getByText('No Citations Available')).toBeInTheDocument();
    });
  });
}); 
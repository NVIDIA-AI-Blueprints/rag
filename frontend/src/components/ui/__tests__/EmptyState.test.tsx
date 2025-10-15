import { describe, it, expect } from 'vitest';
import { render, screen } from '../../../test/utils';
import { EmptyState } from '../EmptyState';

describe('EmptyState', () => {
  describe('Rendering', () => {
    it('renders with required title prop', () => {
      render(<EmptyState title="No items found" />);
      
      expect(screen.getByText('No items found')).toBeInTheDocument();
    });

    it('renders with title and description', () => {
      render(
        <EmptyState 
          title="No documents" 
          description="Upload some documents to get started" 
        />
      );
      
      expect(screen.getByText('No documents')).toBeInTheDocument();
      expect(screen.getByText('Upload some documents to get started')).toBeInTheDocument();
    });

    it('renders without description when not provided', () => {
      render(<EmptyState title="Empty state" />);
      
      expect(screen.getByText('Empty state')).toBeInTheDocument();
      expect(screen.queryByText('Upload some documents to get started')).not.toBeInTheDocument();
    });

    it('renders default icon when no custom icon provided', () => {
      const { container } = render(<EmptyState title="Test" />);
      
      const svg = container.querySelector('svg');
      expect(svg).toBeInTheDocument();
      expect(svg).toHaveClass('w-10', 'h-10', 'text-gray-400');
      expect(svg).toHaveAttribute('fill', 'none');
      expect(svg).toHaveAttribute('stroke', 'currentColor');
      expect(svg).toHaveAttribute('stroke-width', '1.5');
    });

    it('renders custom icon when provided', () => {
      const customIcon = <div data-testid="custom-icon">Custom Icon</div>;
      
      render(<EmptyState title="Test" icon={customIcon} />);
      
      expect(screen.getByTestId('custom-icon')).toBeInTheDocument();
      expect(screen.getByText('Custom Icon')).toBeInTheDocument();
    });
  });

  describe('Styling', () => {
    it('has correct container classes', () => {
      const { container } = render(<EmptyState title="Test" />);
      
      const mainContainer = container.firstChild;
      expect(mainContainer).toHaveClass(
        'flex',
        'h-[300px]',
        'flex-col',
        'items-center',
        'justify-center',
        'text-center',
        'px-4'
      );
    });

    it('has correct icon container classes', () => {
      const { container } = render(<EmptyState title="Test" />);
      
      const iconContainer = container.querySelector('.w-20');
      expect(iconContainer).toHaveClass(
        'w-20',
        'h-20',
        'rounded-full',
        'bg-neutral-800/50',
        'border',
        'border-neutral-700',
        'flex',
        'items-center',
        'justify-center',
        'mb-6',
        'shadow-lg'
      );
    });

    it('has correct title styling', () => {
      render(<EmptyState title="Test Title" />);
      
      const title = screen.getByText('Test Title');
      expect(title).toHaveClass(
        'mb-3',
        'text-xl',
        'font-medium',
        'text-gray-200'
      );
      expect(title.tagName).toBe('H3');
    });

    it('has correct description styling when provided', () => {
      render(<EmptyState title="Test" description="Test description" />);
      
      const description = screen.getByText('Test description');
      expect(description).toHaveClass(
        'text-sm',
        'text-gray-400',
        'max-w-md',
        'leading-relaxed'
      );
      expect(description.tagName).toBe('P');
    });
  });

  describe('Default Icon', () => {
    it('renders correct path for default icon', () => {
      const { container } = render(<EmptyState title="Test" />);
      
      const path = container.querySelector('path');
      expect(path).toBeInTheDocument();
      expect(path).toHaveAttribute('stroke-linecap', 'round');
      expect(path).toHaveAttribute('stroke-linejoin', 'round');
      expect(path).toHaveAttribute('d', 'M19.5 14.25v-2.625a3.375 3.375 0 0 0-3.375-3.375h-1.5A1.125 1.125 0 0 1 13.5 7.125v-1.5a3.375 3.375 0 0 0-3.375-3.375H8.25m2.25 0H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 0 0-9-9Z');
    });
  });

  describe('Props', () => {
    it('handles long titles correctly', () => {
      const longTitle = 'This is a very long title that should still render correctly without breaking the layout or causing issues';
      
      render(<EmptyState title={longTitle} />);
      
      expect(screen.getByText(longTitle)).toBeInTheDocument();
    });

    it('handles long descriptions correctly', () => {
      const longDescription = 'This is a very long description that should wrap properly and maintain good readability while providing detailed information to the user about the current empty state';
      
      render(<EmptyState title="Test" description={longDescription} />);
      
      expect(screen.getByText(longDescription)).toBeInTheDocument();
    });

    it('handles empty string description', () => {
      const { container } = render(<EmptyState title="Test" description="" />);
      
      expect(screen.getByText('Test')).toBeInTheDocument();
      // Empty description should not render the paragraph element - check if there's no paragraph with empty text
      const paragraphs = container.querySelectorAll('p');
      expect(paragraphs.length).toBe(0);
    });

    it('handles React node as custom icon', () => {
      const iconNode = (
        <div>
          <span>Icon</span>
          <span>Text</span>
        </div>
      );
      
      render(<EmptyState title="Test" icon={iconNode} />);
      
      expect(screen.getByText('Icon')).toBeInTheDocument();
      expect(screen.getByText('Text')).toBeInTheDocument();
    });
  });

  describe('Conditional Rendering', () => {
    it('conditionally renders description based on prop', () => {
      const { rerender } = render(<EmptyState title="Test" />);
      
      expect(screen.queryByRole('paragraph')).not.toBeInTheDocument();
      
      rerender(<EmptyState title="Test" description="Now with description" />);
      
      expect(screen.getByText('Now with description')).toBeInTheDocument();
    });
  });
}); 
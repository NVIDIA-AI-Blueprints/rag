import { describe, it, expect } from 'vitest';
import { render } from '../../../test/utils';
import { ExpandChevron } from '../ExpandChevron';

describe('ExpandChevron', () => {
  describe('Rendering', () => {
    it('renders when isExpanded is false', () => {
      const { container } = render(<ExpandChevron isExpanded={false} />);
      
      const chevronContainer = container.firstChild;
      expect(chevronContainer).toBeInTheDocument();
    });

    it('renders when isExpanded is true', () => {
      const { container } = render(<ExpandChevron isExpanded={true} />);
      
      const chevronContainer = container.firstChild;
      expect(chevronContainer).toBeInTheDocument();
    });

    it('renders SVG chevron icon', () => {
      const { container } = render(<ExpandChevron isExpanded={false} />);
      
      const svg = container.querySelector('svg');
      expect(svg).toBeInTheDocument();
      expect(svg).toHaveAttribute('fill', 'none');
      expect(svg).toHaveAttribute('stroke', 'currentColor');
      expect(svg).toHaveAttribute('stroke-width', '2');
      expect(svg).toHaveAttribute('viewBox', '0 0 24 24');
    });

    it('renders the correct path element', () => {
      const { container } = render(<ExpandChevron isExpanded={false} />);
      
      const path = container.querySelector('path');
      expect(path).toBeInTheDocument();
      expect(path).toHaveAttribute('stroke-linecap', 'round');
      expect(path).toHaveAttribute('stroke-linejoin', 'round');
      expect(path).toHaveAttribute('d', 'M9 5l7 7-7 7');
    });
  });

  describe('Styling', () => {
    it('has correct container classes', () => {
      const { container } = render(<ExpandChevron isExpanded={false} />);
      
      const chevronContainer = container.firstChild;
      expect(chevronContainer).toHaveClass(
        'p-1',
        'rounded-md',
        'hover:bg-neutral-800',
        'transition-colors'
      );
    });

    it('has correct base SVG classes', () => {
      const { container } = render(<ExpandChevron isExpanded={false} />);
      
      const svg = container.querySelector('svg');
      expect(svg).toHaveClass(
        'w-4',
        'h-4',
        'text-gray-400',
        'group-hover:text-[var(--nv-green)]',
        'transition-all',
        'duration-200'
      );
    });
  });

  describe('Expansion State', () => {
    it('has rotate-90 class when expanded', () => {
      const { container } = render(<ExpandChevron isExpanded={true} />);
      
      const svg = container.querySelector('svg');
      expect(svg).toHaveClass('rotate-90');
    });

    it('does not have rotate-90 class when not expanded', () => {
      const { container } = render(<ExpandChevron isExpanded={false} />);
      
      const svg = container.querySelector('svg');
      expect(svg).not.toHaveClass('rotate-90');
    });

    it('applies rotation conditionally based on isExpanded prop', () => {
      const { container, rerender } = render(<ExpandChevron isExpanded={false} />);
      
      let svg = container.querySelector('svg');
      expect(svg).not.toHaveClass('rotate-90');
      
      rerender(<ExpandChevron isExpanded={true} />);
      
      svg = container.querySelector('svg');
      expect(svg).toHaveClass('rotate-90');
    });
  });

  describe('Props', () => {
    it('handles isExpanded prop correctly', () => {
      const { container: expandedContainer } = render(<ExpandChevron isExpanded={true} />);
      const { container: collapsedContainer } = render(<ExpandChevron isExpanded={false} />);

      const expandedSvg = expandedContainer.querySelector('svg');
      const collapsedSvg = collapsedContainer.querySelector('svg');

      expect(expandedSvg).toHaveClass('rotate-90');
      expect(collapsedSvg).not.toHaveClass('rotate-90');
    });
  });
}); 
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '../../../test/utils';
import { CollapsibleSection } from '../CollapsibleSection';

describe('CollapsibleSection', () => {
  describe('Expand/Collapse Behavior', () => {
    it('shows content when expanded', () => {
      render(
        <CollapsibleSection title="Test" isExpanded={true} onToggle={vi.fn()}>
          <div data-testid="test-content">Content</div>
        </CollapsibleSection>
      );
      
      expect(screen.getByTestId('test-content')).toBeInTheDocument();
    });

    it('hides content when collapsed', () => {
      render(
        <CollapsibleSection title="Test" isExpanded={false} onToggle={vi.fn()}>
          <div data-testid="test-content">Content</div>
        </CollapsibleSection>
      );
      
      const content = screen.getByTestId('section-content');
      expect(content).toHaveClass('hidden');
    });

    it('calls onToggle when header clicked', () => {
      const onToggle = vi.fn();
      render(
        <CollapsibleSection title="Test" isExpanded={false} onToggle={onToggle}>
          Content
        </CollapsibleSection>
      );
      
      fireEvent.click(screen.getByRole('button'));
      expect(onToggle).toHaveBeenCalledOnce();
    });

    it('shows down chevron when expanded', () => {
      render(<CollapsibleSection title="Test" isExpanded={true} onToggle={vi.fn()}>Content</CollapsibleSection>);
      expect(screen.getByTestId('expand-icon')).toHaveTextContent('▼');
    });

    it('shows right chevron when collapsed', () => {
      render(<CollapsibleSection title="Test" isExpanded={false} onToggle={vi.fn()}>Content</CollapsibleSection>);
      expect(screen.getByTestId('expand-icon')).toHaveTextContent('▶');
    });
  });

  describe('Title Display', () => {
    it('displays the title', () => {
      render(<CollapsibleSection title="My Section" isExpanded={false} onToggle={vi.fn()}>Content</CollapsibleSection>);
      expect(screen.getByText('My Section')).toBeInTheDocument();
    });
  });
}); 
import { describe, it, expect } from 'vitest';
import { render, screen } from '../../../test/utils';
import { StreamingIndicator } from '../StreamingIndicator';

describe('StreamingIndicator', () => {
  describe('Text Display', () => {
    it('displays only dots when no text provided', () => {
      const { container } = render(<StreamingIndicator />);
      
      // Should only show the dots animation, no text - KUI uses Flex components
      const dots = container.querySelectorAll('div[style*="animation"]');
      expect(dots).toHaveLength(3);
    });

    it('displays custom text when provided', () => {
      render(<StreamingIndicator text="Custom" />);
      
      expect(screen.getByText('Custom')).toBeInTheDocument();
    });

    it('displays different custom texts', () => {
      const { rerender } = render(<StreamingIndicator text="Loading" />);
      expect(screen.getByText('Loading')).toBeInTheDocument();
      
      rerender(<StreamingIndicator text="Processing" />);
      expect(screen.getByText('Processing')).toBeInTheDocument();
    });

    it('displays empty text when provided', () => {
      const { container } = render(<StreamingIndicator text="" />);
      
      // Should still show dots - KUI uses Flex components  
      const dots = container.querySelectorAll('div[style*="animation"]');
      expect(dots).toHaveLength(3);
    });

    it('handles special characters in text', () => {
      render(<StreamingIndicator text="Loading... 100%" />);
      expect(screen.getByText('Loading... 100%')).toBeInTheDocument();
    });
  });

  describe('Component Structure', () => {
    it('renders with correct flex container structure', () => {
      const { container } = render(<StreamingIndicator text="Test" />);
      
      // KUI Flex component renders as div, verify basic structure
      expect(container.firstChild).toBeInTheDocument();
      expect(screen.getByText('Test')).toBeInTheDocument();
    });

    it('contains dots animation', () => {
      const { container } = render(<StreamingIndicator />);
      
      // KUI Flex component creates animated dots
      const dots = container.querySelectorAll('div[style*="animation"]');
      expect(dots).toHaveLength(3);
    });
  });
}); 
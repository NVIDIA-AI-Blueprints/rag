import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '../../../test/utils';
import { ErrorState } from '../ErrorState';

describe('ErrorState', () => {
  describe('Message Display', () => {
    it('shows default message when none provided', () => {
      render(<ErrorState />);
      expect(screen.getByText('Something went wrong')).toBeInTheDocument();
    });

    it('shows custom message when provided', () => {
      render(<ErrorState message="Custom error" />);
      expect(screen.getByText('Custom error')).toBeInTheDocument();
    });
  });

  describe('Retry Behavior', () => {
    it('shows no retry button when onRetry not provided', () => {
      render(<ErrorState />);
      expect(screen.queryByText('Try again')).not.toBeInTheDocument();
    });

    it('shows retry button when onRetry provided', () => {
      render(<ErrorState onRetry={vi.fn()} />);
      expect(screen.getByText('Try again')).toBeInTheDocument();
    });

    it('calls onRetry when button clicked', () => {
      const onRetry = vi.fn();
      render(<ErrorState onRetry={onRetry} />);
      
      fireEvent.click(screen.getByText('Try again'));
      expect(onRetry).toHaveBeenCalledOnce();
    });
  });
}); 
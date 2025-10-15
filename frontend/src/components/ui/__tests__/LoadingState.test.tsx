import { describe, it, expect } from 'vitest';
import { render, screen } from '../../../test/utils';
import { LoadingState } from '../LoadingState';

describe('LoadingState', () => {
  it('shows default loading message when none provided', () => {
    render(<LoadingState />);
    expect(screen.getByText('Loading...')).toBeInTheDocument();
  });

  it('shows custom message when provided', () => {
    render(<LoadingState message="Loading data..." />);
    expect(screen.getByText('Loading data...')).toBeInTheDocument();
  });
}); 
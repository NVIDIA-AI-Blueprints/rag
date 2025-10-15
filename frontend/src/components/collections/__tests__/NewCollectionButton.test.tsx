import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '../../../test/utils';
import { NewCollectionButton } from '../NewCollectionButton';

// Mock react-router-dom
const mockNavigate = vi.fn();
vi.mock('react-router-dom', async (importOriginal) => {
  const actual = await importOriginal<typeof import('react-router-dom')>();
  return {
    ...actual,
    useNavigate: () => mockNavigate
  };
});

describe('NewCollectionButton', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Basic Rendering', () => {
    it('renders new collection button', () => {
      render(<NewCollectionButton />);
      
      expect(screen.getByText('+ New Collection')).toBeInTheDocument();
    });

    it('renders as a button element', () => {
      render(<NewCollectionButton />);
      
      expect(screen.getByRole('button')).toBeInTheDocument();
    });
  });

  describe('Navigation Behavior', () => {
    it('navigates to new collection page when clicked', () => {
      render(<NewCollectionButton />);
      
      fireEvent.click(screen.getByText('+ New Collection'));
      
      expect(mockNavigate).toHaveBeenCalledWith('/collections/new');
    });

    it('navigates to correct route on multiple clicks', () => {
      render(<NewCollectionButton />);
      
      const button = screen.getByText('+ New Collection');
      fireEvent.click(button);
      fireEvent.click(button);
      
      expect(mockNavigate).toHaveBeenCalledTimes(2);
      expect(mockNavigate).toHaveBeenNthCalledWith(1, '/collections/new');
      expect(mockNavigate).toHaveBeenNthCalledWith(2, '/collections/new');
    });
  });

  describe('Disabled State', () => {
    it('is enabled by default', () => {
      render(<NewCollectionButton />);
      
      const button = screen.getByRole('button');
      expect(button).not.toBeDisabled();
    });

    it('is disabled when disabled prop is true', () => {
      render(<NewCollectionButton disabled={true} />);
      
      const button = screen.getByRole('button');
      expect(button).toBeDisabled();
    });

    it('is enabled when disabled prop is false', () => {
      render(<NewCollectionButton disabled={false} />);
      
      const button = screen.getByRole('button');
      expect(button).not.toBeDisabled();
    });

    it('does not navigate when disabled and clicked', () => {
      render(<NewCollectionButton disabled={true} />);
      
      fireEvent.click(screen.getByRole('button'));
      
      expect(mockNavigate).not.toHaveBeenCalled();
    });

    it('still navigates when enabled and clicked', () => {
      render(<NewCollectionButton disabled={false} />);
      
      fireEvent.click(screen.getByRole('button'));
      
      expect(mockNavigate).toHaveBeenCalledWith('/collections/new');
    });
  });
}); 
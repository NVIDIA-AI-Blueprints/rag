import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '../../../test/utils';
import { DrawerActions } from '../DrawerActions';

describe('DrawerActions', () => {
  const defaultProps = {
    onDelete: vi.fn(),
    onAddSource: vi.fn(),
    onCloseUploader: vi.fn()
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Button Interactions', () => {
    it('calls onDelete when delete button clicked', () => {
      render(<DrawerActions {...defaultProps} />);
      
      fireEvent.click(screen.getByText('Delete Collection'));
      
      expect(defaultProps.onDelete).toHaveBeenCalledOnce();
    });

    it('calls onAddSource when add source button clicked and uploader is closed', () => {
      render(<DrawerActions {...defaultProps} showUploader={false} />);
      
      fireEvent.click(screen.getByText('Add Source to Collection'));
      
      expect(defaultProps.onAddSource).toHaveBeenCalledOnce();
      expect(defaultProps.onCloseUploader).not.toHaveBeenCalled();
    });

    it('calls onCloseUploader when close uploader button clicked and uploader is open', () => {
      render(<DrawerActions {...defaultProps} showUploader={true} />);
      
      fireEvent.click(screen.getByText('Close Uploader'));
      
      expect(defaultProps.onCloseUploader).toHaveBeenCalledOnce();
      expect(defaultProps.onAddSource).not.toHaveBeenCalled();
    });

    it('does not call onDelete when button is disabled', () => {
      render(<DrawerActions {...defaultProps} isDeleting={true} />);
      
      const deleteButton = screen.getByRole('button', { name: /deleting/i });
      fireEvent.click(deleteButton);
      
      expect(defaultProps.onDelete).not.toHaveBeenCalled();
    });
  });

  describe('Delete Button States', () => {
    it('shows normal delete text when not deleting', () => {
      render(<DrawerActions {...defaultProps} isDeleting={false} />);
      
      expect(screen.getByText('Delete Collection')).toBeInTheDocument();
      expect(screen.queryByText('Deleting...')).not.toBeInTheDocument();
    });

    it('shows deleting text when isDeleting is true', () => {
      render(<DrawerActions {...defaultProps} isDeleting={true} />);
      
      expect(screen.getByText('Deleting...')).toBeInTheDocument();
      expect(screen.queryByText('Delete Collection')).not.toBeInTheDocument();
    });

    it('disables delete button when isDeleting is true', () => {
      render(<DrawerActions {...defaultProps} isDeleting={true} />);
      
      const deleteButton = screen.getByRole('button', { name: /deleting/i });
      expect(deleteButton).toBeDisabled();
    });

    it('enables delete button when isDeleting is false', () => {
      render(<DrawerActions {...defaultProps} isDeleting={false} />);
      
      const deleteButton = screen.getByRole('button', { name: /delete collection/i });
      expect(deleteButton).not.toBeDisabled();
    });

    it('defaults to not deleting when isDeleting prop not provided', () => {
      render(<DrawerActions {...defaultProps} />);
      
      expect(screen.getByText('Delete Collection')).toBeInTheDocument();
      const deleteButton = screen.getByRole('button', { name: /delete collection/i });
      expect(deleteButton).not.toBeDisabled();
    });
  });

  describe('Dynamic Source Button', () => {
    it('shows "Add Source to Collection" text when uploader is closed', () => {
      render(<DrawerActions {...defaultProps} showUploader={false} />);
      
      expect(screen.getByText('Add Source to Collection')).toBeInTheDocument();
      expect(screen.queryByText('Close Uploader')).not.toBeInTheDocument();
    });

    it('shows "Add Source to Collection" text when showUploader prop not provided', () => {
      render(<DrawerActions {...defaultProps} />);
      
      expect(screen.getByText('Add Source to Collection')).toBeInTheDocument();
      expect(screen.queryByText('Close Uploader')).not.toBeInTheDocument();
    });

    it('shows "Close Uploader" text when uploader is open', () => {
      render(<DrawerActions {...defaultProps} showUploader={true} />);
      
      expect(screen.getByText('Close Uploader')).toBeInTheDocument();
      expect(screen.queryByText('Add Source to Collection')).not.toBeInTheDocument();
    });

    it('source button is never disabled even when delete is in progress', () => {
      render(<DrawerActions {...defaultProps} isDeleting={true} showUploader={false} />);
      
      const addButton = screen.getByRole('button', { name: /add source to collection/i });
      expect(addButton).not.toBeDisabled();
    });

    it('close uploader button is never disabled even when delete is in progress', () => {
      render(<DrawerActions {...defaultProps} isDeleting={true} showUploader={true} />);
      
      const closeButton = screen.getByRole('button', { name: /close uploader/i });
      expect(closeButton).not.toBeDisabled();
    });
  });

  describe('Button Presence', () => {
    it('renders both action buttons', () => {
      render(<DrawerActions {...defaultProps} />);
      
      expect(screen.getByRole('button', { name: /delete collection/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /add source to collection/i })).toBeInTheDocument();
    });
  });
}); 
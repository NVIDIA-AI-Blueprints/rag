import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '../../../test/utils';
import { ModalContainer } from '../ModalContainer';
import type React from 'react';

// Mock createPortal to render in place
vi.mock('react-dom', () => ({
  createPortal: (children: React.ReactNode) => children
}));

describe('ModalContainer', () => {
  const mockProps = {
    isOpen: true,
    onClose: vi.fn(),
    title: 'Test Modal',
    children: <div data-testid="modal-content">Modal Content</div>
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Visibility', () => {
    it('renders nothing when closed', () => {
      const { container } = render(<ModalContainer {...mockProps} isOpen={false} />);
      expect(container.firstChild).toBeNull();
    });

    it('renders modal when open', () => {
      render(<ModalContainer {...mockProps} />);
      expect(screen.getByText('Test Modal')).toBeInTheDocument();
      expect(screen.getByTestId('modal-content')).toBeInTheDocument();
    });
  });

  describe('Close Actions', () => {
    it('calls onClose when close button clicked', () => {
      render(<ModalContainer {...mockProps} />);
      fireEvent.click(screen.getByTitle('Close modal'));
      expect(mockProps.onClose).toHaveBeenCalledOnce();
    });

    it('calls onClose when backdrop clicked', () => {
      render(<ModalContainer {...mockProps} />);
      const backdrop = document.querySelector('.bg-black.bg-opacity-60');
      fireEvent.click(backdrop!);
      expect(mockProps.onClose).toHaveBeenCalledOnce();
    });

    it('calls onClose when modal container clicked', () => {
      render(<ModalContainer {...mockProps} />);
      const modalWrapper = document.querySelector('.fixed.inset-0');
      fireEvent.click(modalWrapper!);
      expect(mockProps.onClose).toHaveBeenCalledOnce();
    });
  });

  describe('Content Rendering', () => {
    it('displays correct title', () => {
      render(<ModalContainer {...mockProps} title="Custom Title" />);
      expect(screen.getByText('Custom Title')).toBeInTheDocument();
    });

    it('renders children content', () => {
      const customChildren = <div data-testid="custom-content">Custom Content</div>;
      render(<ModalContainer {...mockProps} children={customChildren} />);
      expect(screen.getByTestId('custom-content')).toBeInTheDocument();
    });

    it('accepts custom maxWidth prop', () => {
      const { container } = render(<ModalContainer {...mockProps} maxWidth="max-w-4xl" />);
      const modalContent = container.querySelector('.max-w-4xl');
      expect(modalContent).toBeInTheDocument();
    });

    it('uses default maxWidth when not provided', () => {
      const { container } = render(<ModalContainer {...mockProps} />);
      const modalContent = container.querySelector('.max-w-2xl');
      expect(modalContent).toBeInTheDocument();
    });
  });
}); 
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '../../../test/utils';
import { FeatureWarningModal } from '../FeatureWarningModal';
import type React from 'react';

// Mock createPortal to render in place
vi.mock('react-dom', () => ({
  createPortal: (children: React.ReactNode) => children
}));

describe('FeatureWarningModal', () => {
  const mockProps = {
    isOpen: true,
    onClose: vi.fn(),
    onConfirm: vi.fn()
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Visibility', () => {
    it('renders nothing when closed', () => {
      const { container } = render(<FeatureWarningModal {...mockProps} isOpen={false} />);
      expect(container.firstChild).toBeNull();
    });

    it('renders modal content when open', () => {
      render(<FeatureWarningModal {...mockProps} />);
      expect(screen.getByText('Feature Requirement')).toBeInTheDocument();
      expect(screen.getByText(/Your model needs to have this feature enabled/)).toBeInTheDocument();
    });
  });

  describe('User Actions', () => {
    it('calls onClose when Cancel button clicked', () => {
      render(<FeatureWarningModal {...mockProps} />);
      fireEvent.click(screen.getByText('Cancel'));
      expect(mockProps.onClose).toHaveBeenCalledOnce();
    });

    // Note: Backdrop click handling is managed by KUI Modal internally via onOpenChange prop

    it('calls onConfirm with false when Enable button clicked without checkbox', () => {
      render(<FeatureWarningModal {...mockProps} />);
      fireEvent.click(screen.getByText('Enable Anyway'));
      expect(mockProps.onConfirm).toHaveBeenCalledWith(false);
    });

    it('calls onConfirm with true when checkbox checked and Enable clicked', () => {
      render(<FeatureWarningModal {...mockProps} />);
      
      const checkbox = screen.getByRole('checkbox');
      fireEvent.click(checkbox);
      fireEvent.click(screen.getByText('Enable Anyway'));
      
      expect(mockProps.onConfirm).toHaveBeenCalledWith(true);
    });
  });

  describe('Checkbox State', () => {
    it('starts with checkbox unchecked', () => {
      render(<FeatureWarningModal {...mockProps} />);
      const checkbox = screen.getByRole('checkbox');
      expect(checkbox).not.toBeChecked();
    });

    it('toggles checkbox state when clicked', () => {
      render(<FeatureWarningModal {...mockProps} />);
      const checkbox = screen.getByRole('checkbox');
      
      fireEvent.click(checkbox);
      expect(checkbox).toBeChecked();
      
      fireEvent.click(checkbox);
      expect(checkbox).not.toBeChecked();
    });

    it('displays correct label for checkbox', () => {
      render(<FeatureWarningModal {...mockProps} />);
      expect(screen.getByText("Don't show this message again")).toBeInTheDocument();
    });
  });
}); 
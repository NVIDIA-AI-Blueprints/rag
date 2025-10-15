import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, fireEvent } from '../../../test/utils';
import { FileInput } from '../FileInput';

// Mock the useFileUpload hook
const mockHandleInputChange = vi.fn();
vi.mock('../../../hooks/useFileUpload', () => ({
  useFileUpload: () => ({
    handleInputChange: mockHandleInputChange
  })
}));

describe('FileInput', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Basic Functionality', () => {
    it('calls handleInputChange when files selected', () => {
      const { container } = render(<FileInput />);
      
      const input = container.querySelector('input[type="file"]');
      const mockFiles = [new File([''], 'test.pdf')];
      
      fireEvent.change(input!, { target: { files: mockFiles } });
      
      expect(mockHandleInputChange).toHaveBeenCalledOnce();
    });

    it('uses multiple prop correctly', () => {
      const { container } = render(<FileInput multiple={false} />);
      
      const input = container.querySelector('input[type="file"]');
      expect(input).not.toHaveAttribute('multiple');
    });

    it('defaults to multiple files when prop not provided', () => {
      const { container } = render(<FileInput />);
      
      const input = container.querySelector('input[type="file"]');
      expect(input).toHaveAttribute('multiple');
    });

    it('applies accept attribute when provided', () => {
      const { container } = render(<FileInput accept=".pdf,.doc" />);
      
      const input = container.querySelector('input[type="file"]');
      expect(input).toHaveAttribute('accept', '.pdf,.doc');
    });

    it('applies custom className when provided', () => {
      const { container } = render(<FileInput className="custom-class" />);
      
      const input = container.querySelector('input[type="file"]');
      expect(input).toHaveClass('custom-class');
    });
  });

  describe('File Type Handling', () => {
    it('renders as file input type', () => {
      const { container } = render(<FileInput />);
      
      const input = container.querySelector('input[type="file"]');
      expect(input).toHaveAttribute('type', 'file');
    });
  });
}); 
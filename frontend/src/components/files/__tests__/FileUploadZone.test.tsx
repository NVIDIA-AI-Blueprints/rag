import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '../../../test/utils';
import { FileUploadZone } from '../FileUploadZone';

// Mock the drag and drop hook
const mockDragHandlers = {
  onDragEnter: vi.fn(),
  onDragLeave: vi.fn(),
  onDragOver: vi.fn(),
  onDrop: vi.fn()
};

vi.mock('../../../hooks/useDragAndDrop', () => ({
  useDragAndDrop: () => ({
    isDragOver: false,
    dragHandlers: mockDragHandlers
  })
}));

describe('FileUploadZone', () => {
  const defaultProps = {
    acceptedTypes: ['.pdf', '.doc'],
    maxFileSize: 50,
    onFilesSelected: vi.fn()
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Basic Rendering', () => {
    it('displays choose files button', () => {
      render(<FileUploadZone {...defaultProps} />);
      expect(screen.getByText('Choose files')).toBeInTheDocument();
    });

    it('displays accepted types and size limit', () => {
      render(<FileUploadZone {...defaultProps} />);
      expect(screen.getByText(/Accepted: \.pdf, \.doc/)).toBeInTheDocument();
      expect(screen.getByText(/Up to 50 MB/)).toBeInTheDocument();
    });

    it('displays drag and drop instruction', () => {
      render(<FileUploadZone {...defaultProps} />);
      expect(screen.getByText(/or drag and drop them here/)).toBeInTheDocument();
    });
  });

  describe('File Selection', () => {
    it('calls onFilesSelected when files chosen through input', () => {
      const { container } = render(<FileUploadZone {...defaultProps} />);
      
      const hiddenInput = container.querySelector('input[type="file"]');
      const mockFiles = [new File([''], 'test.pdf')];
      
      fireEvent.change(hiddenInput!, { target: { files: mockFiles } });
      
      expect(defaultProps.onFilesSelected).toHaveBeenCalledWith(mockFiles);
    });

    it('triggers hidden input when choose files button clicked', () => {
      const mockOnFilesSelected = vi.fn();
      const clickSpy = vi.spyOn(HTMLInputElement.prototype, 'click');
      
      render(
        <FileUploadZone
          acceptedTypes={['.pdf', '.txt']}
          maxFileSize={10}
          onFilesSelected={mockOnFilesSelected}
        />
      );
      
      fireEvent.click(screen.getByText('Choose files'));
      
      // Expect 2 calls now: one from the button, one from the entire area click handler
      expect(clickSpy).toHaveBeenCalledTimes(2);
    });

    it('resets input value after file selection', () => {
      const { container } = render(<FileUploadZone {...defaultProps} />);
      
      const hiddenInput = container.querySelector('input[type="file"]') as HTMLInputElement;
      const mockFiles = [new File([''], 'test.pdf')];
      
      // Set initial value (this simulates user selecting a file)
      Object.defineProperty(hiddenInput, 'value', { writable: true });
      hiddenInput.value = 'test.pdf';
      
      fireEvent.change(hiddenInput, { target: { files: mockFiles } });
      
      expect(hiddenInput.value).toBe('');
    });

    it('does not call onFilesSelected when no files selected', () => {
      const { container } = render(<FileUploadZone {...defaultProps} />);
      
      const hiddenInput = container.querySelector('input[type="file"]');
      
      fireEvent.change(hiddenInput!, { target: { files: [] } });
      
      expect(defaultProps.onFilesSelected).not.toHaveBeenCalled();
    });
  });

  describe('File Input Configuration', () => {
    it('configures hidden input with correct accept attribute', () => {
      const { container } = render(<FileUploadZone {...defaultProps} />);
      
      const hiddenInput = container.querySelector('input[type="file"]');
      expect(hiddenInput).toHaveAttribute('accept', '.pdf,.doc');
    });

    it('configures hidden input as multiple', () => {
      const { container } = render(<FileUploadZone {...defaultProps} />);
      
      const hiddenInput = container.querySelector('input[type="file"]');
      expect(hiddenInput).toHaveAttribute('multiple');
    });

    it('hides file input from view', () => {
      const { container } = render(<FileUploadZone {...defaultProps} />);
      
      const hiddenInput = container.querySelector('input[type="file"]');
      expect(hiddenInput).toHaveClass('hidden');
    });
  });

  describe('Accepted Types Display', () => {
    it('shows different accepted types correctly', () => {
      const props = {
        ...defaultProps,
        acceptedTypes: ['.txt', '.jpg', '.png']
      };
      
      render(<FileUploadZone {...props} />);
      expect(screen.getByText(/Accepted: \.txt, \.jpg, \.png/)).toBeInTheDocument();
    });

    it('shows different max file size correctly', () => {
      const props = {
        ...defaultProps,
        maxFileSize: 100
      };
      
      render(<FileUploadZone {...props} />);
      expect(screen.getByText(/Up to 100 MB/)).toBeInTheDocument();
    });
  });
}); 
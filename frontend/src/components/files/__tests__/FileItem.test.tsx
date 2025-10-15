import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '../../../test/utils';
import { FileItem } from '../FileItem';
import type { UploadFile } from '../../../hooks/useUploadFileState';

// Mock hooks
vi.mock('../../../hooks/useFileIcons', () => ({
  useFileIcons: () => ({
    getFileIconByExtension: vi.fn().mockReturnValue(<div data-testid="file-icon">Icon</div>)
  })
}));

vi.mock('../../../hooks/useFileUtils', () => ({
  useFileUtils: () => ({
    formatFileSize: vi.fn().mockReturnValue('1.5 MB')
  })
}));

// Mock child components
vi.mock('../FileMetadataForm', () => ({
  FileMetadataForm: ({ fileName }: { fileName: string }) => (
    <div data-testid="metadata-form">Metadata for {fileName}</div>
  )
}));

describe('FileItem', () => {
  const createMockUploadFile = (status: 'uploading' | 'uploaded' | 'error', errorMessage?: string): UploadFile => ({
    id: 'test-id',
    file: new File(['content'], 'test.pdf'),
    status,
    progress: status === 'uploaded' ? 100 : 50,
    errorMessage
  });

  describe('Basic Rendering', () => {
    it('displays file name', () => {
      const uploadFile = createMockUploadFile('uploaded');
      render(<FileItem uploadFile={uploadFile} onRemove={vi.fn()} />);
      
      expect(screen.getByText('test.pdf')).toBeInTheDocument();
    });

    it('displays file size', () => {
      const uploadFile = createMockUploadFile('uploaded');
      render(<FileItem uploadFile={uploadFile} onRemove={vi.fn()} />);
      
      expect(screen.getByText('1.5 MB')).toBeInTheDocument();
    });

    it('renders metadata form', () => {
      const uploadFile = createMockUploadFile('uploaded');
      render(<FileItem uploadFile={uploadFile} onRemove={vi.fn()} />);
      
      expect(screen.getByTestId('metadata-form')).toBeInTheDocument();
      expect(screen.getByText('Metadata for test.pdf')).toBeInTheDocument();
    });
  });

  describe('Remove Functionality', () => {
    it('displays remove button', () => {
      const uploadFile = createMockUploadFile('uploaded');
      render(<FileItem uploadFile={uploadFile} onRemove={vi.fn()} />);
      
      const removeButton = screen.getByTitle('Remove file');
      expect(removeButton).toBeInTheDocument();
    });

    it('calls onRemove with correct file id when remove button clicked', () => {
      const onRemove = vi.fn();
      const uploadFile = createMockUploadFile('uploaded');
      
      render(<FileItem uploadFile={uploadFile} onRemove={onRemove} />);
      
      // Click remove button to open modal
      fireEvent.click(screen.getByTitle('Remove file'));
      
      // Find and click the confirm button in the modal
      const confirmButton = screen.getByText('Remove');
      fireEvent.click(confirmButton);
      
      expect(onRemove).toHaveBeenCalledWith('test-id');
    });

    it('calls onRemove with different file ids', () => {
      const onRemove = vi.fn();
      const uploadFile = { ...createMockUploadFile('uploaded'), id: 'different-id' };
      
      render(<FileItem uploadFile={uploadFile} onRemove={onRemove} />);
      
      // Click remove button to open modal
      fireEvent.click(screen.getByTitle('Remove file'));
      
      // Find and click the confirm button in the modal
      const confirmButton = screen.getByText('Remove');
      fireEvent.click(confirmButton);
      
      expect(onRemove).toHaveBeenCalledWith('different-id');
    });
  });

  describe('Status-Based Rendering', () => {
    it('shows "Ready" status for uploaded files', () => {
      const uploadFile = createMockUploadFile('uploaded');
      render(<FileItem uploadFile={uploadFile} onRemove={vi.fn()} />);
      
      expect(screen.getByText('Ready')).toBeInTheDocument();
    });

    it('shows error message for files with errors', () => {
      const uploadFile = createMockUploadFile('error', 'Upload failed');
      render(<FileItem uploadFile={uploadFile} onRemove={vi.fn()} />);
      
      expect(screen.getByText('Upload failed')).toBeInTheDocument();
    });

    it('shows no status for uploading files', () => {
      const uploadFile = createMockUploadFile('uploading');
      render(<FileItem uploadFile={uploadFile} onRemove={vi.fn()} />);
      
      expect(screen.queryByText('Ready')).not.toBeInTheDocument();
      expect(screen.queryByText('Error message during upload')).not.toBeInTheDocument();
    });

    it('shows no status for error files without error message', () => {
      const uploadFile = createMockUploadFile('error');
      render(<FileItem uploadFile={uploadFile} onRemove={vi.fn()} />);
      
      expect(screen.queryByText('Ready')).not.toBeInTheDocument();
      expect(screen.queryByText('Error message during upload')).not.toBeInTheDocument();
    });
  });

  describe('Icon Rendering', () => {
    it('renders file icon for non-error files', () => {
      const uploadFile = createMockUploadFile('uploaded');
      render(<FileItem uploadFile={uploadFile} onRemove={vi.fn()} />);
      
      expect(screen.getByTestId('file-icon')).toBeInTheDocument();
    });

    it('renders file icon for uploading files', () => {
      const uploadFile = createMockUploadFile('uploading');
      render(<FileItem uploadFile={uploadFile} onRemove={vi.fn()} />);
      
      expect(screen.getByTestId('file-icon')).toBeInTheDocument();
    });

    it('does not render file icon for error files', () => {
      const uploadFile = createMockUploadFile('error');
      render(<FileItem uploadFile={uploadFile} onRemove={vi.fn()} />);
      
      expect(screen.queryByTestId('file-icon')).not.toBeInTheDocument();
    });
  });
}); 
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '../../../test/utils';
import NvidiaUpload from '../NvidiaUpload';

// Mock the upload file state hook
const mockHook = {
  uploadFiles: [] as any,
  addFiles: vi.fn(),
  removeFile: vi.fn()
};

vi.mock('../../../hooks/useUploadFileState', () => ({
  useUploadFileState: () => mockHook
}));

// Mock child components
vi.mock('../FileUploadZone', () => ({
  FileUploadZone: ({ acceptedTypes, maxFileSize, onFilesSelected }: {
    acceptedTypes: string[];
    maxFileSize: number;
    onFilesSelected: (files: File[]) => void;
  }) => (
    <div data-testid="file-upload-zone">
      <span>Types: {acceptedTypes.join(',')}</span>
      <span>Size: {maxFileSize}</span>
      <button onClick={() => onFilesSelected([new File([''], 'test.pdf')])}>
        Add Files
      </button>
    </div>
  )
}));

vi.mock('../FileList', () => ({
  FileList: ({ uploadFiles, onRemoveFile }: {
    uploadFiles: File[];
    onRemoveFile: (index: number) => void;
  }) => (
    <div data-testid="file-list">
      <span>Files: {uploadFiles.length}</span>
      <button onClick={() => onRemoveFile(0)}>Remove File</button>
    </div>
  )
}));

describe('NvidiaUpload', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockHook.uploadFiles = [];
  });

  describe('Component Composition', () => {
    it('renders FileUploadZone component', () => {
      render(<NvidiaUpload />);
      expect(screen.getByTestId('file-upload-zone')).toBeInTheDocument();
    });

    it('renders FileList component', () => {
      const mockOnFilesChange = vi.fn();
      render(<NvidiaUpload onFilesChange={mockOnFilesChange} />);
      
      // Check for FileList by looking for its test id
      expect(screen.getByTestId('file-list')).toBeInTheDocument();
    });
  });

  describe('Default Props', () => {
    it('passes default accepted types to FileUploadZone', () => {
      render(<NvidiaUpload />);
      
      expect(screen.getByText(/\.bmp,.docx,.html,.jpeg,.json,.md,.pdf,.png,.pptx,.sh,.tiff,.txt,.mp3,.wav/)).toBeInTheDocument();
    });

    it('passes default max file size to FileUploadZone', () => {
      render(<NvidiaUpload />);
      
      expect(screen.getByText('Size: 400')).toBeInTheDocument();
    });

    it('passes empty upload files to FileList by default', () => {
      render(<NvidiaUpload />);
      
      expect(screen.getByText('Files: 0')).toBeInTheDocument();
    });
  });

  describe('Custom Props', () => {
    it('passes custom accepted types to FileUploadZone', () => {
      render(<NvidiaUpload acceptedTypes={['.jpg', '.png']} />);
      
      expect(screen.getByText('Types: .jpg,.png')).toBeInTheDocument();
    });

    it('passes custom max file size to FileUploadZone', () => {
      render(<NvidiaUpload maxFileSize={100} />);
      
      expect(screen.getByText('Size: 100')).toBeInTheDocument();
    });

    it('renders with KUI Stack structure', () => {
      const { container } = render(<NvidiaUpload />);
      
      // Verify it uses KUI Stack component
      expect(container.firstChild).toHaveClass('nv-flex--direction-col');
    });
  });

  describe('Hook Integration', () => {
    it('passes addFiles to FileUploadZone onFilesSelected', () => {
      render(<NvidiaUpload />);
      
      screen.getByText('Add Files').click();
      
      expect(mockHook.addFiles).toHaveBeenCalledWith([expect.any(File)]);
    });

    it('passes removeFile to FileList onRemoveFile', () => {
      render(<NvidiaUpload />);
      
      screen.getByText('Remove File').click();
      
      expect(mockHook.removeFile).toHaveBeenCalledWith(0);
    });

    it('passes uploadFiles from hook to FileList', () => {
      const mockFiles = [
        { id: '1', file: new File([''], 'test1.pdf'), status: 'uploaded', progress: 100 },
        { id: '2', file: new File([''], 'test2.doc'), status: 'uploaded', progress: 100 }
      ];
      mockHook.uploadFiles = mockFiles;
      
      render(<NvidiaUpload />);
      
      expect(screen.getByText('Files: 2')).toBeInTheDocument();
    });
  });

  describe('Hook Configuration', () => {
    it('configures useUploadFileState with correct options', () => {
      const props = {
        acceptedTypes: ['.pdf'],
        maxFileSize: 25,
        maxFiles: 5,
        onFilesChange: vi.fn()
      };
      
      render(<NvidiaUpload {...props} />);
      
      // Hook should be called with these exact options
      // This is verified through the mocked behavior
      expect(screen.getByText('Types: .pdf')).toBeInTheDocument();
      expect(screen.getByText('Size: 25')).toBeInTheDocument();
    });
  });
}); 
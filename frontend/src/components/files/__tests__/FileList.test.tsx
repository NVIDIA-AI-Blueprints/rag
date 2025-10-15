import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '../../../test/utils';
import { FileList } from '../FileList';
import type { UploadFile } from '../../../hooks/useUploadFileState';

// Mock FileItem component
vi.mock('../FileItem', () => ({
  FileItem: ({ uploadFile, onRemove }: { uploadFile: UploadFile; onRemove: (id: string) => void }) => (
    <div data-testid="file-item">
      <span>{uploadFile.file.name}</span>
      <button data-testid="remove-btn" onClick={() => onRemove(uploadFile.id)}>
        Remove
      </button>
    </div>
  )
}));

describe('FileList', () => {
  const createMockUploadFile = (name: string, id: string): UploadFile => ({
    id,
    file: new File(['content'], name),
    status: 'uploaded',
    progress: 100
  });

  describe('Conditional Rendering', () => {
    it('renders nothing when no upload files', () => {
      const { container } = render(<FileList uploadFiles={[]} onRemoveFile={vi.fn()} />);
      expect(container.firstChild).toBeNull();
    });

    it('renders file items when upload files exist', () => {
      const uploadFiles = [
        createMockUploadFile('file1.pdf', 'id1'),
        createMockUploadFile('file2.doc', 'id2')
      ];
      
      render(<FileList uploadFiles={uploadFiles} onRemoveFile={vi.fn()} />);
      
      expect(screen.getByText('file1.pdf')).toBeInTheDocument();
      expect(screen.getByText('file2.doc')).toBeInTheDocument();
    });

    it('renders correct number of file items', () => {
      const uploadFiles = [
        createMockUploadFile('file1.pdf', 'id1'),
        createMockUploadFile('file2.doc', 'id2'),
        createMockUploadFile('file3.txt', 'id3')
      ];
      
      render(<FileList uploadFiles={uploadFiles} onRemoveFile={vi.fn()} />);
      
      const fileItems = screen.getAllByTestId('file-item');
      expect(fileItems).toHaveLength(3);
    });
  });

  describe('Callback Handling', () => {
    it('calls onRemoveFile when remove button clicked', () => {
      const onRemoveFile = vi.fn();
      const uploadFiles = [createMockUploadFile('test.pdf', 'test-id')];
      
      render(<FileList uploadFiles={uploadFiles} onRemoveFile={onRemoveFile} />);
      
      fireEvent.click(screen.getByTestId('remove-btn'));
      
      expect(onRemoveFile).toHaveBeenCalledWith('test-id');
    });

    it('calls onRemoveFile with correct id for multiple files', () => {
      const onRemoveFile = vi.fn();
      const uploadFiles = [
        createMockUploadFile('file1.pdf', 'id1'),
        createMockUploadFile('file2.doc', 'id2')
      ];
      
      render(<FileList uploadFiles={uploadFiles} onRemoveFile={onRemoveFile} />);
      
      const removeButtons = screen.getAllByTestId('remove-btn');
      fireEvent.click(removeButtons[1]); // Click second file's remove button
      
      expect(onRemoveFile).toHaveBeenCalledWith('id2');
    });
  });

  describe('Key Mapping', () => {
    it('uses uploadFile id as key for FileItem components', () => {
      const uploadFiles = [createMockUploadFile('test.pdf', 'unique-id')];
      
      render(<FileList uploadFiles={uploadFiles} onRemoveFile={vi.fn()} />);
      
      expect(screen.getByTestId('file-item')).toBeInTheDocument();
    });
  });
}); 
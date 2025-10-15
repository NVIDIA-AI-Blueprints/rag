import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '../../../test/utils';
import { FilesList } from '../FilesList';

// Mock the store
const mockStore = {
  selectedFiles: [] as File[]
};

vi.mock('../../../store/useNewCollectionStore', () => ({
  useNewCollectionStore: () => mockStore
}));

// Mock FileCard component
vi.mock('../FileCard', () => ({
  FileCard: ({ file, index }: { file: File; index: number }) => (
    <div data-testid="file-card">
      <span>{file.name}</span>
      <span>Index: {index}</span>
    </div>
  )
}));

describe('FilesList', () => {
  beforeEach(() => {
    mockStore.selectedFiles = [];
  });

  describe('Conditional Rendering', () => {
    it('renders nothing when no files selected', () => {
      mockStore.selectedFiles = [];
      
      const { container } = render(<FilesList />);
      expect(container.firstChild).toBeNull();
    });

    it('renders file cards when files are selected', () => {
      const mockFile1 = new File(['content1'], 'file1.pdf');
      const mockFile2 = new File(['content2'], 'file2.doc');
      mockStore.selectedFiles = [mockFile1, mockFile2];
      
      render(<FilesList />);
      
      expect(screen.getByText('file1.pdf')).toBeInTheDocument();
      expect(screen.getByText('file2.doc')).toBeInTheDocument();
    });

    it('passes correct index to each FileCard', () => {
      const mockFile1 = new File(['content1'], 'first.pdf');
      const mockFile2 = new File(['content2'], 'second.doc');
      mockStore.selectedFiles = [mockFile1, mockFile2];
      
      render(<FilesList />);
      
      expect(screen.getByText('Index: 0')).toBeInTheDocument();
      expect(screen.getByText('Index: 1')).toBeInTheDocument();
    });

    it('uses file name as key for FileCard components', () => {
      const mockFile = new File(['content'], 'unique.pdf');
      mockStore.selectedFiles = [mockFile];
      
      render(<FilesList />);
      
      expect(screen.getByTestId('file-card')).toBeInTheDocument();
    });
  });

  describe('Multiple Files Handling', () => {
    it('renders multiple file cards', () => {
      const files = [
        new File(['1'], 'file1.pdf'),
        new File(['2'], 'file2.doc'),
        new File(['3'], 'file3.txt')
      ];
      mockStore.selectedFiles = files;
      
      render(<FilesList />);
      
      const fileCards = screen.getAllByTestId('file-card');
      expect(fileCards).toHaveLength(3);
    });
  });
}); 
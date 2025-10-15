import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '../../../test/utils';
import { FileCard } from '../FileCard';

// Mock the store
const mockStore = {
  metadataSchema: [] as Array<{ name: string; type: string }>,
  fileMetadata: {} as Record<string, Record<string, unknown>>,
  removeFile: vi.fn(),
  updateMetadataField: vi.fn()
};

vi.mock('../../../store/useNewCollectionStore', () => ({
  useNewCollectionStore: () => mockStore
}));

// Mock MetadataField component
vi.mock('../MetadataField', () => ({
  MetadataField: ({ fileName, field }: { fileName: string; field: { name: string; type: string } }) => (
    <div data-testid="metadata-field">
      {fileName} - {field.name} ({field.type})
    </div>
  )
}));

describe('FileCard', () => {
  const mockFile = new File(['content'], 'test.pdf');

  beforeEach(() => {
    vi.clearAllMocks();
    mockStore.metadataSchema = [];
  });

  describe('File Display', () => {
    it('displays file name', () => {
      render(<FileCard file={mockFile} index={0} />);
      expect(screen.getByText('test.pdf')).toBeInTheDocument();
    });

    it('displays remove button', () => {
      render(<FileCard file={mockFile} index={0} />);
      expect(screen.getByText('REMOVE')).toBeInTheDocument();
    });
  });

  describe('Remove Functionality', () => {
    it('calls removeFile when remove button clicked', () => {
      render(<FileCard file={mockFile} index={2} />);
      
      // Click REMOVE button to open modal
      fireEvent.click(screen.getByText('REMOVE'));
      
      // Click confirm button in modal
      fireEvent.click(screen.getByText('Remove'));
      
      expect(mockStore.removeFile).toHaveBeenCalledWith(2);
    });

    it('calls removeFile with correct index for different files', () => {
      const anotherFile = new File(['content'], 'another.doc');
      render(<FileCard file={anotherFile} index={5} />);
      
      // Click REMOVE button to open modal
      fireEvent.click(screen.getByText('REMOVE'));
      
      // Click confirm button in modal
      fireEvent.click(screen.getByText('Remove'));
      
      expect(mockStore.removeFile).toHaveBeenCalledWith(5);
    });
  });

  describe('Metadata Schema Integration', () => {
    it('hides metadata section when no schema', () => {
      mockStore.metadataSchema = [];
      
      render(<FileCard file={mockFile} index={0} />);
      
      expect(screen.queryByTestId('metadata-field')).not.toBeInTheDocument();
    });

    it('shows metadata fields when schema exists', () => {
      mockStore.metadataSchema = [
        { name: 'author', type: 'string' },
        { name: 'date', type: 'datetime' }
      ];
      
      render(<FileCard file={mockFile} index={0} />);
      
      expect(screen.getByText('test.pdf - author (string)')).toBeInTheDocument();
      expect(screen.getByText('test.pdf - date (datetime)')).toBeInTheDocument();
    });

    it('renders correct number of metadata fields', () => {
      mockStore.metadataSchema = [
        { name: 'field1', type: 'string' },
        { name: 'field2', type: 'number' },
        { name: 'field3', type: 'datetime' }
      ];
      
      render(<FileCard file={mockFile} index={0} />);
      
      const metadataFields = screen.getAllByTestId('metadata-field');
      expect(metadataFields).toHaveLength(3);
    });

    it('passes correct props to MetadataField components', () => {
      mockStore.metadataSchema = [{ name: 'title', type: 'string' }];
      
      render(<FileCard file={mockFile} index={0} />);
      
      expect(screen.getByText('test.pdf - title (string)')).toBeInTheDocument();
    });
  });
}); 
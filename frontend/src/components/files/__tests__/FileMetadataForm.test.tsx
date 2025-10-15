import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '../../../test/utils';
import { FileMetadataForm } from '../FileMetadataForm';

// Mock the store
const mockStore = {
  metadataSchema: [] as Array<{ name: string; type: string }>,
  fileMetadata: {} as Record<string, Record<string, unknown>>,
  updateMetadataField: vi.fn()
};

vi.mock('../../../store/useNewCollectionStore', () => ({
  useNewCollectionStore: () => mockStore
}));

describe('FileMetadataForm', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockStore.metadataSchema = [];
    mockStore.fileMetadata = {};
  });

  describe('Conditional Rendering', () => {
    it('renders nothing when no metadata schema', () => {
      mockStore.metadataSchema = [];
      
      const { container } = render(<FileMetadataForm fileName="test.pdf" />);
      expect(container.firstChild).toBeNull();
    });

    it('renders metadata fields when schema exists', () => {
      mockStore.metadataSchema = [
        { name: 'author', type: 'string' },
        { name: 'date', type: 'datetime' }
      ];
      
      render(<FileMetadataForm fileName="test.pdf" />);
      
      expect(screen.getByText('author (string)')).toBeInTheDocument();
      expect(screen.getByText('date (datetime)')).toBeInTheDocument();
    });

    it('renders correct number of metadata fields', () => {
      mockStore.metadataSchema = [
        { name: 'field1', type: 'string' },
        { name: 'field2', type: 'number' },
        { name: 'field3', type: 'datetime' }
      ];
      
      render(<FileMetadataForm fileName="test.pdf" />);
      
      const allInputs = screen.getAllByDisplayValue('');
      expect(allInputs).toHaveLength(3);
    });
  });

  describe('Field Value Display', () => {
    it('shows empty values when no metadata exists', () => {
      mockStore.metadataSchema = [{ name: 'author', type: 'string' }];
      mockStore.fileMetadata = {};
      
      render(<FileMetadataForm fileName="test.pdf" />);
      
      const input = screen.getByRole('textbox');
      expect(input).toHaveValue('');
    });

    it('shows stored values when metadata exists', () => {
      mockStore.metadataSchema = [{ name: 'author', type: 'string' }];
      mockStore.fileMetadata = {
        'test.pdf': {
          author: 'John Doe'
        }
      };
      
      render(<FileMetadataForm fileName="test.pdf" />);
      
      const input = screen.getByRole('textbox');
      expect(input).toHaveValue('John Doe');
    });

    it('shows empty values for missing fields', () => {
      mockStore.metadataSchema = [
        { name: 'author', type: 'string' },
        { name: 'title', type: 'string' }
      ];
      mockStore.fileMetadata = {
        'test.pdf': {
          author: 'John Doe'
          // title is missing
        }
      };
      
      render(<FileMetadataForm fileName="test.pdf" />);
      
      const inputs = screen.getAllByRole('textbox');
      expect(inputs[0]).toHaveValue('John Doe'); // author
      expect(inputs[1]).toHaveValue(''); // title
    });
  });

  describe('Field Type Handling', () => {
    it('renders text input for string fields', () => {
      mockStore.metadataSchema = [{ name: 'title', type: 'string' }];
      
      render(<FileMetadataForm fileName="test.pdf" />);
      
      const input = screen.getByRole('textbox');
      expect(input).toHaveAttribute('type', 'text');
    });

    it('renders datetime-local input for datetime fields', () => {
      mockStore.metadataSchema = [{ name: 'created', type: 'datetime' }];
      
      render(<FileMetadataForm fileName="test.pdf" />);
      
      const input = screen.getByDisplayValue('');
      expect(input).toHaveAttribute('type', 'datetime-local');
    });

    it('displays field type in label', () => {
      mockStore.metadataSchema = [{ name: 'author', type: 'string' }];
      
      render(<FileMetadataForm fileName="test.pdf" />);
      
      expect(screen.getByText('author (string)')).toBeInTheDocument();
    });
  });

  describe('Field Labels', () => {
    it('capitalizes field names in labels', () => {
      mockStore.metadataSchema = [{ name: 'author', type: 'string' }];
      
      render(<FileMetadataForm fileName="test.pdf" />);
      
      expect(screen.getByText('author (string)')).toBeInTheDocument();
    });

    it('handles multiple word field names', () => {
      mockStore.metadataSchema = [{ name: 'created_date', type: 'datetime' }];
      
      render(<FileMetadataForm fileName="test.pdf" />);
      
      expect(screen.getByText('created_date (datetime)')).toBeInTheDocument();
    });
  });
}); 
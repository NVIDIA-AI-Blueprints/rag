import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '../../../test/utils';
import { CollectionChips } from '../CollectionChips';

// Mock store
const mockStore = {
  selectedCollections: [] as string[],
  toggleCollection: vi.fn()
};

vi.mock('../../../store/useCollectionsStore', () => ({
  useCollectionsStore: () => mockStore
}));

describe('CollectionChips', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockStore.selectedCollections = [];
  });

  describe('Conditional Rendering', () => {
    it('renders nothing when no collections selected', () => {
      mockStore.selectedCollections = [];
      
      const { container } = render(<CollectionChips />);
      
      expect(container.firstChild).toBeNull();
    });

    it('renders chips when collections are selected', () => {
      mockStore.selectedCollections = ['collection1', 'collection2'];
      
      render(<CollectionChips />);
      
      expect(screen.getByText('collection1')).toBeInTheDocument();
      expect(screen.getByText('collection2')).toBeInTheDocument();
    });

    it('renders Collections label when chips are shown', () => {
      mockStore.selectedCollections = ['test-collection'];
      
      render(<CollectionChips />);
      
      expect(screen.getByText('Collections:')).toBeInTheDocument();
    });

    it('renders correct number of chips', () => {
      mockStore.selectedCollections = ['col1', 'col2', 'col3'];
      
      render(<CollectionChips />);
      
      expect(screen.getByText('col1')).toBeInTheDocument();
      expect(screen.getByText('col2')).toBeInTheDocument();
      expect(screen.getByText('col3')).toBeInTheDocument();
    });
  });

  describe('Remove Functionality', () => {
    it('calls toggleCollection when remove button clicked', () => {
      mockStore.selectedCollections = ['test-collection'];
      
      render(<CollectionChips />);
      
      const removeButton = screen.getByTitle('Remove collection');
      fireEvent.click(removeButton);
      
      expect(mockStore.toggleCollection).toHaveBeenCalledWith('test-collection');
    });

    it('calls toggleCollection with correct collection name', () => {
      mockStore.selectedCollections = ['collection-a', 'collection-b'];
      
      render(<CollectionChips />);
      
      const removeButtons = screen.getAllByTitle('Remove collection');
      fireEvent.click(removeButtons[1]); // Click second collection's remove button
      
      expect(mockStore.toggleCollection).toHaveBeenCalledWith('collection-b');
    });

    it('each chip has its own remove button', () => {
      mockStore.selectedCollections = ['col1', 'col2'];
      
      render(<CollectionChips />);
      
      const removeButtons = screen.getAllByTitle('Remove collection');
      expect(removeButtons).toHaveLength(2);
    });

    it('remove buttons are clickable', () => {
      mockStore.selectedCollections = ['test'];
      
      render(<CollectionChips />);
      
      const removeButton = screen.getByTitle('Remove collection');
      expect(removeButton).toBeInTheDocument();
      
      fireEvent.click(removeButton);
      expect(mockStore.toggleCollection).toHaveBeenCalledOnce();
    });
  });

  describe('Collection Names Display', () => {
    it('displays collection names correctly', () => {
      mockStore.selectedCollections = ['my-collection', 'another-one'];
      
      render(<CollectionChips />);
      
      expect(screen.getByText('my-collection')).toBeInTheDocument();
      expect(screen.getByText('another-one')).toBeInTheDocument();
    });

    it('handles special characters in collection names', () => {
      mockStore.selectedCollections = ['collection_with_underscores', 'collection-with-dashes'];
      
      render(<CollectionChips />);
      
      expect(screen.getByText('collection_with_underscores')).toBeInTheDocument();
      expect(screen.getByText('collection-with-dashes')).toBeInTheDocument();
    });

    it('handles single collection', () => {
      mockStore.selectedCollections = ['single-collection'];
      
      render(<CollectionChips />);
      
      expect(screen.getByText('single-collection')).toBeInTheDocument();
      expect(screen.getByText('Collections:')).toBeInTheDocument();
    });
  });
}); 
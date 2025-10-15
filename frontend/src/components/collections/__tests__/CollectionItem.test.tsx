import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '../../../test/utils';
import { CollectionItem } from '../CollectionItem';
import type { MetadataFieldType } from '../../../types/collections';

// Mock stores
const mockCollectionsStore = {
  selectedCollections: [] as string[],
  toggleCollection: vi.fn()
};

const mockDrawerStore = {
  openDrawer: vi.fn()
};

const mockNotificationStore = {
  getPendingTasks: vi.fn().mockReturnValue([])
};

vi.mock('../../../store/useCollectionsStore', () => ({
  useCollectionsStore: () => mockCollectionsStore
}));

vi.mock('../../../store/useCollectionDrawerStore', () => ({
  useCollectionDrawerStore: () => mockDrawerStore
}));

vi.mock('../../../store/useNotificationStore', () => ({
  useNotificationStore: () => mockNotificationStore
}));

describe('CollectionItem', () => {
  const mockCollection = {
    collection_name: 'test-collection',
    num_entities: 100,
    metadata_schema: []
  };

  beforeEach(() => {
    vi.clearAllMocks();
    mockCollectionsStore.selectedCollections = [];
    mockNotificationStore.getPendingTasks.mockReturnValue([]);
  });

  describe('Basic Rendering', () => {
    it('displays collection name', () => {
      render(<CollectionItem collection={mockCollection} />);
      
      expect(screen.getByText('test-collection')).toBeInTheDocument();
    });

    it('renders with KUI components', () => {
      const { container } = render(<CollectionItem collection={mockCollection} />);
      
      // KUI Flex component renders as div, check for basic structure
      expect(container.firstChild).toBeInTheDocument();
      expect(screen.getByText('test-collection')).toBeInTheDocument();
    });
  });

  describe('Content Display', () => {
    it('displays collection information correctly', () => {
      render(<CollectionItem collection={mockCollection} />);
      
      expect(screen.getByText('test-collection')).toBeInTheDocument();
      expect(screen.getByText('100 entities')).toBeInTheDocument();
    });

    it('shows correct entity count for different collections', () => {
      const collection2 = { ...mockCollection, collection_name: 'other-collection', num_entities: 250 };
      render(<CollectionItem collection={collection2} />);
      
      expect(screen.getByText('other-collection')).toBeInTheDocument();
      expect(screen.getByText('250 entities')).toBeInTheDocument();
    });
  });

  describe('Drawer Opening', () => {
    it('calls openDrawer when more button clicked', () => {
      render(<CollectionItem collection={mockCollection} />);
      
      const moreButton = screen.getByRole('button');
      fireEvent.click(moreButton);
      
      expect(mockDrawerStore.openDrawer).toHaveBeenCalledWith(mockCollection);
    });

    it('does not call toggleCollection when more button clicked', () => {
      render(<CollectionItem collection={mockCollection} />);
      
      const moreButton = screen.getByRole('button');
      fireEvent.click(moreButton);
      
      expect(mockCollectionsStore.toggleCollection).not.toHaveBeenCalled();
    });

    it('passes correct collection to openDrawer', () => {
      const customCollection = {
        collection_name: 'custom-collection',
        num_entities: 50,
        metadata_schema: [{ name: 'field1', type: 'string' as MetadataFieldType, description: 'desc' }]
      };
      
      render(<CollectionItem collection={customCollection} />);
      
      const moreButton = screen.getByRole('button');
      fireEvent.click(moreButton);
      
      expect(mockDrawerStore.openDrawer).toHaveBeenCalledWith(customCollection);
    });
  });

  describe('Pending Tasks Display', () => {
    it('shows more button when no pending tasks', () => {
      mockNotificationStore.getPendingTasks.mockReturnValue([]);
      
      render(<CollectionItem collection={mockCollection} />);
      
      expect(screen.getByRole('button')).toBeInTheDocument();
    });

    it('hides more button when collection has pending tasks', () => {
      mockNotificationStore.getPendingTasks.mockReturnValue([{
        collection_name: 'test-collection',
        state: 'PENDING'
      }]);
      
      render(<CollectionItem collection={mockCollection} />);
      
      expect(screen.queryByRole('button')).not.toBeInTheDocument();
    });

    it('shows spinner when collection has pending tasks', () => {
      mockNotificationStore.getPendingTasks.mockReturnValue([{
        collection_name: 'test-collection',
        state: 'PENDING'
      }]);
      
      const { container } = render(<CollectionItem collection={mockCollection} />);
      
      const spinner = container.querySelector('.animate-spin');
      expect(spinner).toBeInTheDocument();
    });

    it('shows more button for collection without pending tasks while other has pending', () => {
      mockNotificationStore.getPendingTasks.mockReturnValue([{
        collection_name: 'other-collection',
        state: 'PENDING'
      }]);
      
      render(<CollectionItem collection={mockCollection} />);
      
      expect(screen.getByRole('button')).toBeInTheDocument();
    });

    it('ignores non-pending tasks', () => {
      mockNotificationStore.getPendingTasks.mockReturnValue([{
        collection_name: 'test-collection',
        state: 'FINISHED'
      }]);
      
      render(<CollectionItem collection={mockCollection} />);
      
      expect(screen.getByRole('button')).toBeInTheDocument();
    });
  });
}); 
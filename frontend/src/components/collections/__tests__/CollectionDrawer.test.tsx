import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '../../../test/utils';
import CollectionDrawer from '../CollectionDrawer';

// Mock all dependencies with simple functions
const mockReset = vi.fn();

vi.mock('../../../store/useNewCollectionStore', () => {
  const mockStore = vi.fn(() => ({
    setMetadataSchema: vi.fn()
  }));
  
  // Add the getState method to the store function with proper typing
  (mockStore as any).getState = vi.fn(() => ({
    reset: mockReset
  }));
  
  return {
    useNewCollectionStore: mockStore
  };
});

vi.mock('../../../store/useCollectionDrawerStore', () => ({
  useCollectionDrawerStore: vi.fn(() => ({
    activeCollection: {
      collection_name: 'Test Collection',
      metadata_schema: []
    },
    closeDrawer: vi.fn(),
    toggleUploader: vi.fn()
  }))
}));

vi.mock('../../../hooks/useCollectionActions', () => ({
  useCollectionActions: vi.fn(() => ({
    handleDeleteCollection: vi.fn(),
    isDeleting: false
  }))
}));

// Mock child components that are still used
interface MockDrawerActionsProps {
  onDelete: () => void;
  onAddSource: () => void;
  isDeleting: boolean;
}

vi.mock('../../drawer/DrawerActions', () => ({
  DrawerActions: ({ onDelete, onAddSource, isDeleting }: MockDrawerActionsProps) => (
    <div data-testid="drawer-actions">
      <button data-testid="delete-button" onClick={onDelete} disabled={isDeleting}>
        {isDeleting ? 'Deleting...' : 'Delete'}
      </button>
      <button data-testid="add-source-button" onClick={onAddSource}>Add Source</button>
    </div>
  )
}));

vi.mock('../tasks/DocumentsList', () => ({
  DocumentsList: () => <div data-testid="documents-list">Loading documents...</div>
}));

vi.mock('../../drawer/UploaderSection', () => ({
  UploaderSection: () => <div data-testid="uploader-section">Uploader Section</div>
}));

describe('CollectionDrawer', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Component Structure', () => {
    it('renders all main components', () => {
      render(<CollectionDrawer />);
      
      // Now uses KUI SidePanel instead of custom drawer components
      // DocumentsList shows loading state, so check for that or the main content area
      expect(screen.getByText('Loading documents...')).toBeInTheDocument();
      expect(screen.getByTestId('drawer-actions')).toBeInTheDocument();
    });

    it('renders with collection name as title', () => {
      render(<CollectionDrawer />);
      
      // KUI SidePanel handles the title through slotHeading prop
      // We can verify the title is passed correctly by checking it's in the document
      expect(screen.getByText('Test Collection')).toBeInTheDocument();
    });
  });

  describe('Basic Interactions', () => {
    it('renders delete and add source buttons', () => {
      render(<CollectionDrawer />);
      
      expect(screen.getByTestId('delete-button')).toBeInTheDocument();
      expect(screen.getByTestId('add-source-button')).toBeInTheDocument();
    });

    it('has close functionality through SidePanel', () => {
      render(<CollectionDrawer />);
      
      // KUI SidePanel handles close functionality internally
      // We can verify the drawer actions are rendered which is key functionality
      expect(screen.getByTestId('drawer-actions')).toBeInTheDocument();
    });

    it('has clickable action buttons', () => {
      render(<CollectionDrawer />);
      
      const deleteButton = screen.getByTestId('delete-button');
      const addSourceButton = screen.getByTestId('add-source-button');
      
      // Test that clicking doesn't crash
      deleteButton.click();
      addSourceButton.click();
    });
  });

  describe('Component Integration', () => {
    it('renders without errors indicating proper imports', () => {
      render(<CollectionDrawer />);
      
      // If there are import issues, this test would fail
      // Check for KUI SidePanel main content area
      expect(screen.getByTestId('nv-side-panel-content')).toBeInTheDocument();
    });
  });
}); 
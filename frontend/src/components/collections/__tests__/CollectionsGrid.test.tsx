import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '../../../test/utils';
import { CollectionsGrid } from '../CollectionsGrid';

// Mock API hook
const mockUseCollections = {
  data: null as any,
  isLoading: false,
  error: null as any
};

vi.mock('../../../api/useCollectionsApi', () => ({
  useCollections: () => mockUseCollections
}));

// Mock child components
vi.mock('../CollectionDrawer', () => ({
  LoadingState: ({ message }: { message: string }) => <div data-testid="loading-state">{message}</div>,
  ErrorState: ({ message, onRetry }: { message: string; onRetry: () => void }) => (
    <div data-testid="error-state">
      <span>{message}</span>
      <button onClick={onRetry}>Retry</button>
    </div>
  ),
  EmptyState: ({ title, description }: { title: string; description: string }) => (
    <div data-testid="empty-state">
      <span>{title}</span>
      <span>{description}</span>
    </div>
  )
}));

vi.mock('../CollectionItem', () => ({
  CollectionItem: ({ collection }: { collection: { collection_name: string } }) => (
    <div data-testid="collection-item">{collection.collection_name}</div>
  )
}));

describe('CollectionsGrid', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUseCollections.data = null;
    mockUseCollections.isLoading = false;
    mockUseCollections.error = null;
  });

  describe('Loading State', () => {
    it('shows loading state when isLoading is true', () => {
      mockUseCollections.isLoading = true;
      
      render(<CollectionsGrid searchQuery="" />);
      
      // KUI Spinner has different data-testid
      expect(screen.getByTestId('nv-spinner-spinner')).toBeInTheDocument();
      expect(screen.getByText('Loading collections...')).toBeInTheDocument();
    });

    it('does not show loading state when isLoading is false', () => {
      mockUseCollections.isLoading = false;
      mockUseCollections.data = [];
      
      render(<CollectionsGrid searchQuery="" />);
      
      expect(screen.queryByTestId('nv-spinner-spinner')).not.toBeInTheDocument();
    });
  });

  describe('Error State', () => {
    it('shows error state when error exists', () => {
      mockUseCollections.error = new Error('Failed to fetch');
      
      render(<CollectionsGrid searchQuery="" />);
      
      // KUI StatusMessage has different data-testid
      expect(screen.getByTestId('nv-status-message-root')).toBeInTheDocument();
      expect(screen.getByText('Failed to load collections')).toBeInTheDocument();
    });

    it('shows error message without retry button', () => {
      mockUseCollections.error = new Error('Network error');
      
      render(<CollectionsGrid searchQuery="" />);
      
      // KUI error state doesn't include a retry button, just shows the error message
      expect(screen.getByTestId('nv-status-message-root')).toBeInTheDocument();
      expect(screen.getByText('Failed to load collections')).toBeInTheDocument();
      
      // No retry button in KUI implementation
      expect(screen.queryByText('Retry')).not.toBeInTheDocument();
    });

    it('does not show error state when no error', () => {
      mockUseCollections.error = null;
      mockUseCollections.data = [];
      
      render(<CollectionsGrid searchQuery="" />);
      
      expect(screen.queryByTestId('error-state')).not.toBeInTheDocument();
    });
  });

  describe('Empty States', () => {
    it('shows empty state when no collections', () => {
      mockUseCollections.data = [];
      
      render(<CollectionsGrid searchQuery="" />);
      
      // KUI StatusMessage used for empty state
      expect(screen.getByTestId('nv-status-message-root')).toBeInTheDocument();
      expect(screen.getByText('No collections')).toBeInTheDocument();
    });

    it('shows search empty state when no search results', () => {
      mockUseCollections.data = [{ collection_name: 'test-collection' }];
      
      render(<CollectionsGrid searchQuery="nonexistent" />);
      
      // KUI StatusMessage used for search empty state
      expect(screen.getByTestId('nv-status-message-root')).toBeInTheDocument();
      expect(screen.getByText('No matches found')).toBeInTheDocument();
      expect(screen.getByText('No collections match "nonexistent"')).toBeInTheDocument();
    });

    it('shows collections when search matches', () => {
      mockUseCollections.data = [{ collection_name: 'test-collection' }];
      
      render(<CollectionsGrid searchQuery="test" />);
      
      // Should not show empty state StatusMessage when collections are found
      expect(screen.queryByText('No matches found')).not.toBeInTheDocument();
      expect(screen.getByTestId('collection-item')).toBeInTheDocument();
    });
  });

  describe('Search Filtering', () => {
    it('filters collections by search query (case insensitive)', () => {
      mockUseCollections.data = [
        { collection_name: 'first-collection' },
        { collection_name: 'second-collection' },
        { collection_name: 'third-set' }
      ];
      
      render(<CollectionsGrid searchQuery="COLLECTION" />);
      
      expect(screen.getByText('first-collection')).toBeInTheDocument();
      expect(screen.getByText('second-collection')).toBeInTheDocument();
      expect(screen.queryByText('third-set')).not.toBeInTheDocument();
    });

    it('shows all collections when search query is empty', () => {
      mockUseCollections.data = [
        { collection_name: 'collection1' },
        { collection_name: 'collection2' }
      ];
      
      render(<CollectionsGrid searchQuery="" />);
      
      expect(screen.getByText('collection1')).toBeInTheDocument();
      expect(screen.getByText('collection2')).toBeInTheDocument();
    });

    it('handles partial matches', () => {
      mockUseCollections.data = [
        { collection_name: 'user-data' },
        { collection_name: 'user-profiles' },
        { collection_name: 'system-logs' }
      ];
      
      render(<CollectionsGrid searchQuery="user" />);
      
      expect(screen.getByText('user-data')).toBeInTheDocument();
      expect(screen.getByText('user-profiles')).toBeInTheDocument();
      expect(screen.queryByText('system-logs')).not.toBeInTheDocument();
    });

    it('returns no results for non-matching search', () => {
      mockUseCollections.data = [
        { collection_name: 'collection1' },
        { collection_name: 'collection2' }
      ];
      
      render(<CollectionsGrid searchQuery="xyz" />);
      
      expect(screen.queryByText('collection1')).not.toBeInTheDocument();
      expect(screen.queryByText('collection2')).not.toBeInTheDocument();
      expect(screen.getByText('No matches found')).toBeInTheDocument();
    });
  });

  describe('Collections Rendering', () => {
    it('renders collection items when data is available', () => {
      mockUseCollections.data = [
        { collection_name: 'test1' },
        { collection_name: 'test2' }
      ];
      
      render(<CollectionsGrid searchQuery="" />);
      
      const items = screen.getAllByTestId('collection-item');
      expect(items).toHaveLength(2);
      expect(screen.getByText('test1')).toBeInTheDocument();
      expect(screen.getByText('test2')).toBeInTheDocument();
    });
  });
}); 
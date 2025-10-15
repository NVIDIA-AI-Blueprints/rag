import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '../../../test/utils';
import StatusMessages from '../StatusMessages';

// Mock the store
const mockStore = {
  collectionName: '',
  error: null as string | null,
  uploadComplete: false
};

vi.mock('../../../store/useNewCollectionStore', () => ({
  useNewCollectionStore: () => mockStore
}));

describe('StatusMessages', () => {
  beforeEach(() => {
    // Reset mock store to default state
    mockStore.collectionName = '';
    mockStore.error = null;
    mockStore.uploadComplete = false;
  });

  describe('Error Messages', () => {
    it('shows error message when error exists', () => {
      mockStore.error = 'Something went wrong';
      
      render(<StatusMessages />);
      expect(screen.getByText('Something went wrong')).toBeInTheDocument();
    });

    it('hides error message when no error', () => {
      mockStore.error = null;
      
      render(<StatusMessages />);
      expect(screen.queryByText(/went wrong/)).not.toBeInTheDocument();
    });

    it('displays different error messages', () => {
      mockStore.error = 'Custom error message';
      
      render(<StatusMessages />);
      expect(screen.getByText('Custom error message')).toBeInTheDocument();
    });
  });

  describe('Success Messages', () => {
    it('shows success message when upload complete', () => {
      mockStore.uploadComplete = true;
      mockStore.collectionName = 'test-collection';
      
      render(<StatusMessages />);
      expect(screen.getByText('Collection "test-collection" created successfully.')).toBeInTheDocument();
    });

    it('hides success message when upload not complete', () => {
      mockStore.uploadComplete = false;
      mockStore.collectionName = 'test-collection';
      
      render(<StatusMessages />);
      expect(screen.queryByText(/created successfully/)).not.toBeInTheDocument();
    });

    it('includes collection name in success message', () => {
      mockStore.uploadComplete = true;
      mockStore.collectionName = 'my-custom-collection';
      
      render(<StatusMessages />);
      expect(screen.getByText('Collection "my-custom-collection" created successfully.')).toBeInTheDocument();
    });
  });

  describe('Multiple States', () => {
    it('can show both error and success messages', () => {
      mockStore.error = 'Warning message';
      mockStore.uploadComplete = true;
      mockStore.collectionName = 'test-collection';
      
      render(<StatusMessages />);
      
      expect(screen.getByText('Warning message')).toBeInTheDocument();
      expect(screen.getByText('Collection "test-collection" created successfully.')).toBeInTheDocument();
    });

    it('shows nothing when no error or success', () => {
      const { container } = render(<StatusMessages />);
      expect(container.firstChild).toBeNull();
    });
  });
}); 
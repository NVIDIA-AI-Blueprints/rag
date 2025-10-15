import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '../../../test/utils';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { DocumentItem } from '../DocumentItem';
import { useCollectionDrawerStore } from '../../../store/useCollectionDrawerStore';

// Mock the file icon hook
vi.mock('../../../hooks/useFileIcons', () => ({
  useFileIcons: () => ({
    getFileIconByExtension: vi.fn().mockReturnValue(<div data-testid="file-icon" />)
  })
}));

const mockSetDeleteError = vi.fn();

vi.mock('../../../store/useCollectionDrawerStore', () => ({
  useCollectionDrawerStore: vi.fn(),
}));

function createQueryClient() {
  return new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
}

describe('DocumentItem', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // Provide mock for all tests
    vi.mocked(useCollectionDrawerStore).mockReturnValue({
      isOpen: false,
      activeCollection: null,
      showUploader: false,
      deleteError: null,
      openDrawer: vi.fn(),
      closeDrawer: vi.fn(),
      toggleUploader: vi.fn(),
      setDeleteError: mockSetDeleteError,
      reset: vi.fn(),
    });
  });

  describe('Basic Rendering', () => {
    it('displays document name', () => {
      const qc = createQueryClient();
      render(
        <QueryClientProvider client={qc}>
          <DocumentItem name="test.pdf" metadata={{}} collectionName="test" />
        </QueryClientProvider>
      );
      expect(screen.getByText('test.pdf')).toBeInTheDocument();
    });

    it('renders file icon', () => {
      const qc = createQueryClient();
      render(
        <QueryClientProvider client={qc}>
          <DocumentItem name="test.pdf" metadata={{}} collectionName="test" />
        </QueryClientProvider>
      );
      expect(screen.getByTestId('file-icon')).toBeInTheDocument();
    });
  });

  describe('Metadata Display', () => {
    it('shows metadata when present', () => {
      const qc = createQueryClient();
      const metadata = { author: 'John Doe', pages: '10' };
      render(
        <QueryClientProvider client={qc}>
          <DocumentItem name="test.pdf" metadata={metadata} collectionName="test" />
        </QueryClientProvider>
      );
      
      expect(screen.getByText('author:')).toBeInTheDocument();
      expect(screen.getByText('John Doe')).toBeInTheDocument();
      expect(screen.getByText('pages:')).toBeInTheDocument();
      expect(screen.getByText('10')).toBeInTheDocument();
    });

    it('hides metadata section when no metadata', () => {
      const qc = createQueryClient();
      render(
        <QueryClientProvider client={qc}>
          <DocumentItem name="test.pdf" metadata={{}} collectionName="test" />
        </QueryClientProvider>
      );
      expect(screen.queryByTestId('document-metadata')).not.toBeInTheDocument();
    });

    it('filters out filename from metadata display', () => {
      const qc = createQueryClient();
      const metadata = { filename: 'test.pdf', author: 'John Doe' };
      render(
        <QueryClientProvider client={qc}>
          <DocumentItem name="test.pdf" metadata={metadata} collectionName="test" />
        </QueryClientProvider>
      );
      
      expect(screen.queryByText('filename:')).not.toBeInTheDocument();
      expect(screen.getByText('author:')).toBeInTheDocument();
    });

    it('hides metadata section when only filename present', () => {
      const qc = createQueryClient();
      const metadata = { filename: 'test.pdf' };
      render(
        <QueryClientProvider client={qc}>
          <DocumentItem name="test.pdf" metadata={metadata} collectionName="test" />
        </QueryClientProvider>
      );
      
      expect(screen.queryByTestId('document-metadata')).not.toBeInTheDocument();
    });
  });

  describe('Delete Button', () => {
    const defaultProps = {
      name: 'test-document.pdf',
      metadata: { category: 'tech' },
      collectionName: 'test-collection',
    };

    it('renders trash icon delete button', () => {
      const qc = createQueryClient();
      
      render(
        <QueryClientProvider client={qc}>
          <DocumentItem {...defaultProps} />
        </QueryClientProvider>
      );

      const deleteButton = screen.getByRole('button', { name: /delete test-document\.pdf/i });
      expect(deleteButton).toBeInTheDocument();
      // The title attribute should be present
      expect(deleteButton).toHaveAttribute('title', 'Delete');
    });

    it('calls delete API when trash icon is clicked and confirmed', async () => {
      const user = userEvent.setup();
      const qc = createQueryClient();

      const fetchMock = vi.spyOn(global, 'fetch');
      fetchMock.mockResolvedValueOnce(
        new Response(JSON.stringify({ message: 'Document deleted' }), { status: 200 })
      );

      render(
        <QueryClientProvider client={qc}>
          <DocumentItem {...defaultProps} />
        </QueryClientProvider>
      );

      const deleteButton = screen.getByRole('button', { name: /delete test-document\.pdf/i });
      await user.click(deleteButton);

      // Find and click the confirm button in the modal
      const confirmButton = screen.getByText('Delete');
      await user.click(confirmButton);

      await waitFor(() => {
        expect(fetchMock).toHaveBeenCalledWith(
          '/api/documents?collection_name=test-collection',
          expect.objectContaining({
            method: 'DELETE',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(['test-document.pdf']),
          })
        );
      });

      expect(mockSetDeleteError).toHaveBeenCalledWith(null);
    });

    it('does not call delete API when confirmation is cancelled', async () => {
      const user = userEvent.setup();
      const qc = createQueryClient();

      const fetchMock = vi.spyOn(global, 'fetch');

      render(
        <QueryClientProvider client={qc}>
          <DocumentItem {...defaultProps} />
        </QueryClientProvider>
      );

      const deleteButton = screen.getByRole('button', { name: /delete test-document\.pdf/i });
      await user.click(deleteButton);

      // Find and click the cancel button in the modal
      const cancelButton = screen.getByText('Cancel');
      await user.click(cancelButton);

      expect(fetchMock).not.toHaveBeenCalled();
    });

    it('does not call delete when collectionName is empty', async () => {
      const user = userEvent.setup();
      const qc = createQueryClient();

      const fetchMock = vi.spyOn(global, 'fetch');

      render(
        <QueryClientProvider client={qc}>
          <DocumentItem {...defaultProps} collectionName="" />
        </QueryClientProvider>
      );

      const deleteButton = screen.getByRole('button', { name: /delete test-document\.pdf/i });
      await user.click(deleteButton);

      expect(fetchMock).not.toHaveBeenCalled();
    });
  });
}); 
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '../../../test/utils';
import { DocumentsList } from '../DocumentsList';

// Mock the hooks at module level
const mockRefetch = vi.fn();
const mockUseCollectionDocuments = vi.fn();

vi.mock('../../../api/useCollectionDocuments', () => ({
  useCollectionDocuments: () => mockUseCollectionDocuments(),
  useDeleteDocument: () => ({ mutate: vi.fn(), isPending: false }),
  useUpdateDocumentMetadata: () => ({ mutate: vi.fn(), isPending: false })
}));

vi.mock('../../../store/useCollectionDrawerStore', () => ({
  useCollectionDrawerStore: () => ({
    activeCollection: { collection_name: 'test-collection' }
  })
}));

describe('DocumentsList', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockRefetch.mockClear();
  });

  it('shows loading state when loading', () => {
    mockUseCollectionDocuments.mockReturnValue({
      data: undefined,
      isLoading: true,
      error: null,
      refetch: mockRefetch
    });

    render(<DocumentsList />);
    expect(screen.getByText('Loading documents...')).toBeInTheDocument();
  });

  it('shows error state when error occurs', () => {
    mockUseCollectionDocuments.mockReturnValue({
      data: undefined,
      isLoading: false,
      error: new Error('Failed to fetch'),
      refetch: mockRefetch
    });

    render(<DocumentsList />);
    expect(screen.getByText('Failed to load documents')).toBeInTheDocument();
  });

  it('shows empty state when no documents', () => {
    mockUseCollectionDocuments.mockReturnValue({
      data: {
        message: 'Success',
        total_documents: 0,
        documents: []
      },
      isLoading: false,
      error: null,
      refetch: mockRefetch
    });

    render(<DocumentsList />);
    expect(screen.getByText('No documents yet')).toBeInTheDocument();
  });

  it('renders documents when data available', () => {
    mockUseCollectionDocuments.mockReturnValue({
      data: {
        message: 'Success',
        total_documents: 2,
        documents: [
          { document_name: 'doc1.pdf', metadata: {} },
          { document_name: 'doc2.txt', metadata: {} }
        ]
      },
      isLoading: false,
      error: null,
      refetch: mockRefetch
    });

    render(<DocumentsList />);
    expect(screen.getByText('doc1.pdf')).toBeInTheDocument();
    expect(screen.getByText('doc2.txt')).toBeInTheDocument();
  });
});

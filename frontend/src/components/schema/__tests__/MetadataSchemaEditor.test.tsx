import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '../../../test/utils';
import MetadataSchemaEditor from '../MetadataSchemaEditor';
import { useSchemaEditor } from '../../../hooks/useSchemaEditor';

// Mock the schema editor hook using working pattern
vi.mock('../../../hooks/useSchemaEditor', () => ({
  useSchemaEditor: vi.fn()
}));

const mockUseSchemaEditor = vi.mocked(useSchemaEditor);

describe('MetadataSchemaEditor', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUseSchemaEditor.mockReturnValue({
      metadataSchema: [],
      showSchemaEditor: false,
      editingIndex: null,
      editValues: {},
      fieldNameError: null,
      setFieldNameError: vi.fn(),
      toggleEditor: vi.fn(),
      startEditing: vi.fn(),
      updateEditValue: vi.fn(),
      commitUpdate: vi.fn(),
      cancelEdit: vi.fn(),
      deleteField: vi.fn()
    });
  });

  describe('Basic Rendering', () => {
    it('renders without crashing', () => {
      expect(() => render(<MetadataSchemaEditor />)).not.toThrow();
    });

    it('renders header with correct text', () => {
      render(<MetadataSchemaEditor />);
      
      expect(screen.getByText('Metadata Schema')).toBeInTheDocument();
    });

    it('renders with correct container styling', () => {
      const { container } = render(<MetadataSchemaEditor />);
      
      const mainContainer = container.firstChild as HTMLElement;
      expect(mainContainer).toHaveClass('nv-panel');
    });

    it('does not render schema content when closed by default', () => {
      render(<MetadataSchemaEditor />);
      
      // Content should not be visible when showSchemaEditor is false (default)
      expect(screen.queryByText('Schema Configuration')).not.toBeInTheDocument();
      expect(screen.queryByText('Define metadata fields for this collection.')).not.toBeInTheDocument();
    });
  });
}); 
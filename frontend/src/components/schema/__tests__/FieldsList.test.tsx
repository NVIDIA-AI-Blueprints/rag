import { describe, it, expect, vi } from 'vitest';
import { render } from '../../../test/utils';
import { FieldsList } from '../FieldsList';

// Mock the schema editor hook using working pattern
vi.mock('../../hooks/useSchemaEditor', () => ({
  useSchemaEditor: () => ({
    metadataSchema: [],
    editingIndex: -1,
    editValues: {},
    startEditing: vi.fn(),
    updateEditValue: vi.fn(),
    commitUpdate: vi.fn(),
    cancelEdit: vi.fn(),
    deleteField: vi.fn()
  })
}));

describe('FieldsList', () => {
  describe('Basic Rendering', () => {
    it('renders without crashing', () => {
      expect(() => render(<FieldsList />)).not.toThrow();
    });

    it('renders with correct container styling', () => {
      const { container } = render(<FieldsList />);
      
      const fieldsList = container.querySelector('.space-y-3');
      expect(fieldsList).toBeInTheDocument();
    });

    it('renders empty list when no fields', () => {
      render(<FieldsList />);
      
      // Should render container but no child elements when empty
      const container = document.querySelector('.space-y-3');
      expect(container).toBeInTheDocument();
      expect(container?.children).toHaveLength(0);
    });
  });
}); 
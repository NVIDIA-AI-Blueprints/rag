import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '../../../test/utils';
import { MetadataField } from '../MetadataField';
import type { UIMetadataField } from '../../../types/collections';

describe('MetadataField', () => {
  const mockOnChange = vi.fn();

  const createMockField = (overrides: Partial<UIMetadataField> = {}): UIMetadataField => ({
    name: 'author',
    type: 'string',
    required: false,
    ...overrides
  });

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('String Fields', () => {
    it('renders string field with correct label', () => {
      const field = createMockField({ name: 'title', type: 'string' });
      
      render(
        <MetadataField
          fileName="test.pdf"
          field={field}
          value=""
          onChange={mockOnChange}
        />
      );
      
      expect(screen.getByText('title (string)')).toBeInTheDocument();
    });

    it('shows required indicator for required fields', () => {
      const field = createMockField({ name: 'author', type: 'string', required: true });
      
      render(
        <MetadataField
          fileName="test.pdf"
          field={field}
          value=""
          onChange={mockOnChange}
        />
      );
      
      expect(screen.getByText('author * (string)')).toBeInTheDocument();
    });

    it('displays field description when provided', () => {
      const field = createMockField({ 
        name: 'category', 
        type: 'string',
        description: 'Document category' 
      });
      
      render(
        <MetadataField
          fileName="test.pdf"
          field={field}
          value=""
          onChange={mockOnChange}
        />
      );
      
      expect(screen.getByText('Document category')).toBeInTheDocument();
    });

    it('displays current value in input', () => {
      const field = createMockField({ name: 'author', type: 'string' });
      
      render(
        <MetadataField
          fileName="test.pdf"
          field={field}
          value="John Doe"
          onChange={mockOnChange}
        />
      );
      
      const input = screen.getByRole('textbox');
      expect(input).toHaveValue('John Doe');
    });

    it('calls onChange when value changes', () => {
      const field = createMockField({ name: 'author', type: 'string' });
      
      render(
        <MetadataField
          fileName="test.pdf"
          field={field}
          value=""
          onChange={mockOnChange}
        />
      );
      
      const input = screen.getByRole('textbox');
      fireEvent.change(input, { target: { value: 'Jane Doe' } });
      
      expect(mockOnChange).toHaveBeenCalledWith('author', 'Jane Doe', 'string');
    });
  });

  describe('Boolean Fields', () => {
    it('renders switch for boolean fields', () => {
      const field = createMockField({ name: 'is_public', type: 'boolean' });
      
      render(
        <MetadataField
          fileName="test.pdf"
          field={field}
          value={false}
          onChange={mockOnChange}
        />
      );
      
      // KUI Switch renders with role="switch"
      const switchElement = screen.getByRole('switch');
      expect(switchElement).toBeInTheDocument();
    });

    it('shows Yes/No text based on value', () => {
      const field = createMockField({ name: 'is_public', type: 'boolean' });
      
      const { rerender } = render(
        <MetadataField
          fileName="test.pdf"
          field={field}
          value={false}
          onChange={mockOnChange}
        />
      );
      
      expect(screen.getByText('No')).toBeInTheDocument();
      
      rerender(
        <MetadataField
          fileName="test.pdf"
          field={field}
          value={true}
          onChange={mockOnChange}
        />
      );
      
      expect(screen.getByText('Yes')).toBeInTheDocument();
    });

    it('calls onChange when switch is toggled', () => {
      const field = createMockField({ name: 'is_public', type: 'boolean' });
      
      render(
        <MetadataField
          fileName="test.pdf"
          field={field}
          value={false}
          onChange={mockOnChange}
        />
      );
      
      const switchElement = screen.getByRole('switch');
      fireEvent.click(switchElement);
      
      expect(mockOnChange).toHaveBeenCalledWith('is_public', true, 'boolean');
    });
  });

  describe('Numeric Fields', () => {
    it('renders input for integer fields', () => {
      const field = createMockField({ name: 'priority', type: 'integer' });
      
      render(
        <MetadataField
          fileName="test.pdf"
          field={field}
          value="5"
          onChange={mockOnChange}
        />
      );
      
      const input = screen.getByDisplayValue('5');
      expect(input).toBeInTheDocument();
    });

    it('renders input for float fields', () => {
      const field = createMockField({ name: 'rating', type: 'float' });
      
      render(
        <MetadataField
          fileName="test.pdf"
          field={field}
          value="4.5"
          onChange={mockOnChange}
        />
      );
      
      const input = screen.getByDisplayValue('4.5');
      expect(input).toBeInTheDocument();
    });
  });

  describe('Datetime Fields', () => {
    it('renders input for datetime fields', () => {
      const field = createMockField({ name: 'created_date', type: 'datetime' });
      
      render(
        <MetadataField
          fileName="test.pdf"
          field={field}
          value=""
          onChange={mockOnChange}
        />
      );
      
      const input = screen.getByRole('textbox');
      expect(input).toBeInTheDocument();
    });

    it('handles datetime value changes', () => {
      const field = createMockField({ name: 'created_date', type: 'datetime' });
      
      render(
        <MetadataField
          fileName="test.pdf"
          field={field}
          value="2024-01-15T10:30"
          onChange={mockOnChange}
        />
      );
      
      const input = screen.getByRole('textbox');
      fireEvent.change(input, { target: { value: '2024-01-15T11:30' } });
      
      expect(mockOnChange).toHaveBeenCalledWith('created_date', '2024-01-15T11:30:00', 'datetime');
    });
  });

  describe('Array Fields', () => {
    it('renders array field with add input', () => {
      const field = createMockField({ 
        name: 'tags', 
        type: 'array',
        array_type: 'string'
      });
      
      render(
        <MetadataField
          fileName="test.pdf"
          field={field}
          value="[]"
          onChange={mockOnChange}
        />
      );
      
      expect(screen.getByText('tags (array<string>)')).toBeInTheDocument();
      expect(screen.getByPlaceholderText('Enter string value')).toBeInTheDocument();
      expect(screen.getByTitle('Add item')).toBeInTheDocument();
    });

    it('displays existing array items', () => {
      const field = createMockField({ 
        name: 'tags', 
        type: 'array',
        array_type: 'string'
      });
      
      render(
        <MetadataField
          fileName="test.pdf"
          field={field}
          value='["tag1", "tag2"]'
          onChange={mockOnChange}
        />
      );
      
      expect(screen.getByText('tag1')).toBeInTheDocument();
      expect(screen.getByText('tag2')).toBeInTheDocument();
    });
  });

  describe('Field Labels', () => {
    it('shows max length info in help text when provided', () => {
      const field = createMockField({ 
        name: 'title', 
        type: 'string',
        max_length: 100
      });
      
      render(
        <MetadataField
          fileName="test.pdf"
          field={field}
          value="Test title"
          onChange={mockOnChange}
        />
      );
      
      expect(screen.getByText('Max 100 characters')).toBeInTheDocument();
    });
  });
});

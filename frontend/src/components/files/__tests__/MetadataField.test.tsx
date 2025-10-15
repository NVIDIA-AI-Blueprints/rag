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
    it('renders checkbox for boolean fields', () => {
      const field = createMockField({ name: 'is_public', type: 'boolean' });
      
      render(
        <MetadataField
          fileName="test.pdf"
          field={field}
          value={false}
          onChange={mockOnChange}
        />
      );
      
      const checkbox = screen.getByRole('checkbox');
      expect(checkbox).toBeInTheDocument();
      expect(checkbox).not.toBeChecked();
    });

    it('shows checked state for true values', () => {
      const field = createMockField({ name: 'is_public', type: 'boolean' });
      
      render(
        <MetadataField
          fileName="test.pdf"
          field={field}
          value={true}
          onChange={mockOnChange}
        />
      );
      
      const checkbox = screen.getByRole('checkbox');
      expect(checkbox).toBeChecked();
    });

    it('calls onChange when checkbox is toggled', () => {
      const field = createMockField({ name: 'is_public', type: 'boolean' });
      
      render(
        <MetadataField
          fileName="test.pdf"
          field={field}
          value={false}
          onChange={mockOnChange}
        />
      );
      
      const checkbox = screen.getByRole('checkbox');
      fireEvent.click(checkbox);
      
      expect(mockOnChange).toHaveBeenCalledWith('is_public', true, 'boolean');
    });
  });

  describe('Numeric Fields', () => {
    it('renders number input for integer fields', () => {
      const field = createMockField({ name: 'priority', type: 'integer' });
      
      render(
        <MetadataField
          fileName="test.pdf"
          field={field}
          value="5"
          onChange={mockOnChange}
        />
      );
      
      const input = screen.getByRole('spinbutton');
      expect(input).toHaveAttribute('type', 'number');
      expect(input).toHaveAttribute('step', '1');
    });

    it('renders number input for float fields with decimal step', () => {
      const field = createMockField({ name: 'rating', type: 'float' });
      
      render(
        <MetadataField
          fileName="test.pdf"
          field={field}
          value="4.5"
          onChange={mockOnChange}
        />
      );
      
      const input = screen.getByRole('spinbutton');
      expect(input).toHaveAttribute('step', '0.01');
    });
  });

  describe('Datetime Fields', () => {
    it('renders datetime-local input for datetime fields', () => {
      const field = createMockField({ name: 'created_date', type: 'datetime' });
      
      render(
        <MetadataField
          fileName="test.pdf"
          field={field}
          value=""
          onChange={mockOnChange}
        />
      );
      
      const input = screen.getByDisplayValue('');
      expect(input).toHaveAttribute('type', 'datetime-local');
    });

    it('formats datetime value correctly', () => {
      const field = createMockField({ name: 'created_date', type: 'datetime' });
      
      render(
        <MetadataField
          fileName="test.pdf"
          field={field}
          value="2024-01-15T10:30"
          onChange={mockOnChange}
        />
      );
      
      const input = screen.getByDisplayValue('2024-01-15T10:30');
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

    // TODO: Implement item count display for arrays with max_length
    // it('shows item count when max_length is set', () => {
    //   const field = createMockField({ 
    //     name: 'tags', 
    //     type: 'array',
    //     array_type: 'string',
    //     max_length: 5
    //   });
    //   
    //   render(
    //     <MetadataField
    //       fileName="test.pdf"
    //       field={field}
    //       value='["tag1", "tag2"]'
    //       onChange={mockOnChange}
    //     />
    //   );
    //   
    //   expect(screen.getByText('2/5 items')).toBeInTheDocument();
    // });
  });

  describe('Validation and Limits', () => {
    it('shows character count for string fields with max_length', () => {
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
      
      expect(screen.getByText('10/100 characters')).toBeInTheDocument();
    });

    it('applies maxLength attribute to string inputs', () => {
      const field = createMockField({ 
        name: 'title', 
        type: 'string',
        max_length: 50
      });
      
      render(
        <MetadataField
          fileName="test.pdf"
          field={field}
          value=""
          onChange={mockOnChange}
        />
      );
      
      const input = screen.getByRole('textbox');
      expect(input).toHaveAttribute('maxLength', '50');
    });
  });
}); 
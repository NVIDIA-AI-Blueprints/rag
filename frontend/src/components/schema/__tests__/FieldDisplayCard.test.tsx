import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '../../../test/utils';
import { FieldDisplayCard } from '../FieldDisplayCard';
import type { UIMetadataField } from '../../../types/collections';

describe('FieldDisplayCard', () => {
  const mockField: UIMetadataField = {
    name: 'test_field',
    type: 'string',
    optional: false
  };

  it('displays field name and type', () => {
    render(<FieldDisplayCard field={mockField} onEdit={vi.fn()} onDelete={vi.fn()} />);
    
    expect(screen.getByText('test_field')).toBeInTheDocument();
    expect(screen.getByText('string')).toBeInTheDocument();
  });

  it('calls onEdit when edit button clicked', () => {
    const onEdit = vi.fn();
    render(<FieldDisplayCard field={mockField} onEdit={onEdit} onDelete={vi.fn()} />);
    
    fireEvent.click(screen.getByTestId('edit-button'));
    expect(onEdit).toHaveBeenCalledOnce();
  });

  it('calls onDelete when delete button clicked', () => {
    const onDelete = vi.fn();
    render(<FieldDisplayCard field={mockField} onEdit={vi.fn()} onDelete={onDelete} />);
    
    fireEvent.click(screen.getByTestId('delete-button'));
    expect(onDelete).toHaveBeenCalledOnce();
  });

  it('handles invalid field type gracefully and shows default color', () => {
    // Mock console.warn to verify warning is logged
    const consoleWarnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    
    const invalidField = {
      ...mockField,
      type: 'invalid_type' as never // Force invalid type for testing
    };
    
    render(<FieldDisplayCard field={invalidField} onEdit={vi.fn()} onDelete={vi.fn()} />);
    
    // Field should still render with the invalid type name displayed
    expect(screen.getByText('test_field')).toBeInTheDocument();
    expect(screen.getByText('invalid_type')).toBeInTheDocument();
    
    // Warning should be logged
    expect(consoleWarnSpy).toHaveBeenCalledWith(
      'Invalid field type encountered: "invalid_type" for field "test_field". Using default color.'
    );
    
    // Badge should still be rendered (with default gray color)
    expect(screen.getByTestId('field-type')).toBeInTheDocument();
    
    // Cleanup
    consoleWarnSpy.mockRestore();
  });
}); 
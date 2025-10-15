import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '../../../test/utils';
import { FieldEditForm } from '../FieldEditForm';
import type { UIMetadataField } from '../../../types/collections';

describe('FieldEditForm', () => {
  const mockField: UIMetadataField = {
    name: 'test-field',
    type: 'string'
  };

  const mockOnUpdate = vi.fn();
  const mockOnSave = vi.fn();
  const mockOnCancel = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Rendering', () => {
    it('renders form with correct structure', () => {
      render(
        <FieldEditForm
          field={mockField}
          onUpdate={mockOnUpdate}
          onSave={mockOnSave}
          onCancel={mockOnCancel}
        />
      );

      expect(screen.getByText('Editing Field')).toBeInTheDocument();
      expect(screen.getByDisplayValue('test-field')).toBeInTheDocument();
      
      // Check the select trigger shows the current value
      const selectTrigger = screen.getByRole('combobox');
      expect(selectTrigger).toHaveTextContent('string');
      
      expect(screen.getByText('Cancel')).toBeInTheDocument();
      expect(screen.getByText('Save Changes')).toBeInTheDocument();
    });

    it('displays current field values', () => {
      render(
        <FieldEditForm
          field={mockField}
          onUpdate={mockOnUpdate}
          onSave={mockOnSave}
          onCancel={mockOnCancel}
        />
      );

      const nameInput = screen.getByDisplayValue('test-field') as HTMLInputElement;
      const selectTrigger = screen.getByRole('combobox');

      expect(nameInput.value).toBe('test-field');
      expect(selectTrigger).toHaveTextContent('string');
    });

    it('handles empty field name', () => {
      const emptyField = { ...mockField, name: '' };
      
      render(
        <FieldEditForm
          field={emptyField}
          onUpdate={mockOnUpdate}
          onSave={mockOnSave}
          onCancel={mockOnCancel}
        />
      );

      const nameInput = screen.getByRole('textbox') as HTMLInputElement;
      expect(nameInput.value).toBe('');
    });

    it('defaults to string type when no type specified', () => {
      const fieldWithoutType = { ...mockField, type: undefined as never };
      
      render(
        <FieldEditForm
          field={fieldWithoutType}
          onUpdate={mockOnUpdate}
          onSave={mockOnSave}
          onCancel={mockOnCancel}
        />
      );

      const selectTrigger = screen.getByRole('combobox');
      expect(selectTrigger).toHaveTextContent('string');
    });
  });

  describe('Form Interactions', () => {
    it('calls onUpdate when field name changes', () => {
      render(
        <FieldEditForm
          field={mockField}
          onUpdate={mockOnUpdate}
          onSave={mockOnSave}
          onCancel={mockOnCancel}
        />
      );

      const nameInput = screen.getByDisplayValue('test-field');
      fireEvent.change(nameInput, { target: { value: 'new-field-name' } });

      expect(mockOnUpdate).toHaveBeenCalledWith({ name: 'new-field-name' });
    });

    it('calls onUpdate when field type changes', () => {
      render(
        <FieldEditForm
          field={mockField}
          onUpdate={mockOnUpdate}
          onSave={mockOnSave}
          onCancel={mockOnCancel}
        />
      );

      // Click the select trigger to open the dropdown
      const selectTrigger = screen.getByRole('combobox');
      fireEvent.click(selectTrigger);

      // Click on a different option
      const datetimeOption = screen.getByRole('option', { name: 'datetime' });
      fireEvent.click(datetimeOption);

      expect(mockOnUpdate).toHaveBeenCalledWith({ type: 'datetime', array_type: undefined });
    });

    it('calls onSave when Save Changes button is clicked', () => {
      render(
        <FieldEditForm
          field={mockField}
          onUpdate={mockOnUpdate}
          onSave={mockOnSave}
          onCancel={mockOnCancel}
        />
      );

      const saveButton = screen.getByText('Save Changes');
      fireEvent.click(saveButton);

      expect(mockOnSave).toHaveBeenCalled();
    });

    it('calls onCancel when Cancel button is clicked', () => {
      render(
        <FieldEditForm
          field={mockField}
          onUpdate={mockOnUpdate}
          onSave={mockOnSave}
          onCancel={mockOnCancel}
        />
      );

      const cancelButton = screen.getByText('Cancel');
      fireEvent.click(cancelButton);

      expect(mockOnCancel).toHaveBeenCalled();
    });

    it('calls onSave when Enter is pressed in name input', () => {
      render(
        <FieldEditForm
          field={mockField}
          onUpdate={mockOnUpdate}
          onSave={mockOnSave}
          onCancel={mockOnCancel}
        />
      );

      const nameInput = screen.getByDisplayValue('test-field');
      fireEvent.keyDown(nameInput, { key: 'Enter' });

      expect(mockOnSave).toHaveBeenCalled();
    });

    it('does not call onSave for other keys', () => {
      render(
        <FieldEditForm
          field={mockField}
          onUpdate={mockOnUpdate}
          onSave={mockOnSave}
          onCancel={mockOnCancel}
        />
      );

      const nameInput = screen.getByDisplayValue('test-field');
      fireEvent.keyDown(nameInput, { key: 'Escape' });

      expect(mockOnSave).not.toHaveBeenCalled();
    });
  });

  describe('Form Styling', () => {
    it('applies correct styling classes', () => {
      render(
        <FieldEditForm
          field={mockField}
          onUpdate={mockOnUpdate}
          onSave={mockOnSave}
          onCancel={mockOnCancel}
        />
      );

      // Check for KUI Block component structure
      const formContainer = document.querySelector('.nv-block');
      expect(formContainer).toBeInTheDocument();
    });

    it('renders all available type options', () => {
      render(
        <FieldEditForm
          field={mockField}
          onUpdate={mockOnUpdate}
          onSave={mockOnSave}
          onCancel={mockOnCancel}
        />
      );

      // Open the select dropdown first
      const selectTrigger = screen.getByRole('combobox');
      fireEvent.click(selectTrigger);

      // Now check for options
      expect(screen.getByRole('option', { name: 'string' })).toBeInTheDocument();
      expect(screen.getByRole('option', { name: 'datetime' })).toBeInTheDocument();
    });
  });

  describe('Edge Cases', () => {
    it('handles null field name', () => {
      const fieldWithNullName = { ...mockField, name: null as unknown as string };
      
      render(
        <FieldEditForm
          field={fieldWithNullName}
          onUpdate={mockOnUpdate}
          onSave={mockOnSave}
          onCancel={mockOnCancel}
        />
      );

      const nameInput = screen.getByRole('textbox') as HTMLInputElement;
      expect(nameInput.value).toBe('');
    });

    it('handles datetime field type', () => {
      const datetimeField = { ...mockField, type: 'datetime' as const };
      
      render(
        <FieldEditForm
          field={datetimeField}
          onUpdate={mockOnUpdate}
          onSave={mockOnSave}
          onCancel={mockOnCancel}
        />
      );

      const selectTrigger = screen.getByRole('combobox');
      expect(selectTrigger).toHaveTextContent('datetime');
    });
  });
}); 
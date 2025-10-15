import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '../../../test/utils';
import { NewFieldForm } from '../NewFieldForm';

// Mock the new field form hook using working pattern
vi.mock('../../hooks/useNewFieldForm', () => ({
  useNewFieldForm: () => ({
    newField: { name: '', type: 'string' },
    fieldNameError: null,
    updateNewField: vi.fn(),
    validateAndAdd: vi.fn(),
    canAdd: false
  })
}));

describe('NewFieldForm', () => {
  describe('Basic Rendering', () => {
    it('renders without crashing', () => {
      expect(() => render(<NewFieldForm />)).not.toThrow();
    });

    it('renders main form elements', () => {
      render(<NewFieldForm />);
      
      // Check for basic form elements - using actual placeholder text
      expect(screen.getByPlaceholderText('e.g. category, author, department')).toBeInTheDocument();
      expect(screen.getByRole('combobox')).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /add field/i })).toBeInTheDocument();
    });

    it('renders with correct container styling', () => {
      const { container } = render(<NewFieldForm />);
      
      const form = container.querySelector('.nv-block');
      expect(form).toBeInTheDocument();
    });

    it('disables add button by default when canAdd is false', () => {
      render(<NewFieldForm />);
      
      const addButton = screen.getByRole('button', { name: /add field/i });
      expect(addButton).toBeDisabled();
    });

    it('does not display error when fieldNameError is null', () => {
      render(<NewFieldForm />);
      
      // Check that no error messages are displayed
      expect(screen.queryByText(/field name is required/i)).not.toBeInTheDocument();
      expect(screen.queryByText(/invalid/i)).not.toBeInTheDocument();
    });
  });
}); 
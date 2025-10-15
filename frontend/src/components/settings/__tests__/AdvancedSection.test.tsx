import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '../../../test/utils';
import userEvent from '@testing-library/user-event';
import { AdvancedSection } from '../AdvancedSection';

// Mock the settings store
const mockUseSettingsStore = vi.fn();

vi.mock('../../../store/useSettingsStore', () => ({
  useSettingsStore: () => mockUseSettingsStore()
}));

describe('AdvancedSection', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUseSettingsStore.mockReturnValue({
      useLocalStorage: false,
      stopTokens: [],
      set: vi.fn()
    });
  });

  describe('Rendering', () => {
    it('renders local storage toggle', () => {
      render(<AdvancedSection />);
      
      expect(screen.getByText('Use Local Storage')).toBeInTheDocument();
      expect(screen.getByText('Disabled')).toBeInTheDocument();
    });

    it('renders stop tokens label', () => {
      render(<AdvancedSection />);
      
      expect(screen.getByText('Stop Tokens')).toBeInTheDocument();
    });

    it('renders help text', () => {
      render(<AdvancedSection />);
      
      expect(screen.getByText('Add text tokens that will stop text generation when encountered. Click on a token to remove it.')).toBeInTheDocument();
    });

    it('renders input with placeholder', () => {
      render(<AdvancedSection />);
      
      const input = screen.getByPlaceholderText('Enter stop token');
      expect(input).toBeInTheDocument();
    });

    it('renders add button', () => {
      render(<AdvancedSection />);
      
      const addButton = screen.getByRole('button', { name: 'Add' });
      expect(addButton).toBeInTheDocument();
      expect(addButton).toBeDisabled(); // Should be disabled when input is empty
    });
  });

  describe('Stop Tokens Functionality', () => {
    it('displays existing stop tokens', () => {
      // Clear any previous mock and set new one
      mockUseSettingsStore.mockClear();
      mockUseSettingsStore.mockReturnValue({
        useLocalStorage: false,
        stopTokens: ['AA', 'BB', 'CC'],
        set: vi.fn()
      });

      render(<AdvancedSection />);
      
      expect(screen.getByText('AA')).toBeInTheDocument();
      expect(screen.getByText('BB')).toBeInTheDocument();
      expect(screen.getByText('CC')).toBeInTheDocument();
    });

    it('allows adding new tokens', async () => {
      const user = userEvent.setup();
      const mockSet = vi.fn();
      mockUseSettingsStore.mockClear();
      mockUseSettingsStore.mockReturnValue({
        useLocalStorage: false,
        stopTokens: [],
        set: mockSet
      });

      render(<AdvancedSection />);
      
      const input = screen.getByPlaceholderText('Enter stop token');
      const addButton = screen.getByRole('button', { name: 'Add' });
      
      // Type into the input
      await user.type(input, 'STOP');
      
      // Click add button
      await user.click(addButton);
      
      expect(mockSet).toHaveBeenCalledWith({ stopTokens: ['STOP'] });
    });

    it('allows any text characters', () => {
      render(<AdvancedSection />);
      
      const input = screen.getByPlaceholderText('Enter stop token');
      
      fireEvent.change(input, { target: { value: 'Hello World!' } });
      
      // Should accept all characters
      expect((input as HTMLInputElement).value).toBe('Hello World!');
    });

    it('allows removing tokens by clicking them', async () => {
      const user = userEvent.setup();
      const mockSet = vi.fn();
      mockUseSettingsStore.mockClear();
      mockUseSettingsStore.mockReturnValue({
        useLocalStorage: false,
        stopTokens: ['FF', 'BB'],
        set: mockSet
      });

      render(<AdvancedSection />);
      
      // Find the tag button by its accessible name (based on the DOM output, it should be "FF")
      const tagButton = screen.getByRole('button', { name: 'FF' });
      await user.click(tagButton);
      
      expect(mockSet).toHaveBeenCalledWith({ stopTokens: ['BB'] });
    });
  });

  describe('Local Storage Toggle', () => {
    it('toggles local storage setting', async () => {
      const user = userEvent.setup();
      const mockSet = vi.fn();
      mockUseSettingsStore.mockClear();
      mockUseSettingsStore.mockReturnValue({
        useLocalStorage: false,
        stopTokens: [],
        set: mockSet
      });

      render(<AdvancedSection />);
      
      const toggle = screen.getByRole('switch');
      await user.click(toggle);
      
      expect(mockSet).toHaveBeenCalledWith({ useLocalStorage: true });
    });
  });


}); 
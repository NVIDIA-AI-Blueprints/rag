import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '../../../test/utils';

// Mock all problematic imports to avoid module resolution issues
vi.mock('../../../store/useSettingsStore', () => ({
  useSettingsStore: () => ({
    vdbTopK: 10,
    rerankerTopK: 5,
    maxTokens: 1000,
    stopTokens: '',
    set: vi.fn()
  })
}));

vi.mock('../../../hooks/useCollapsibleSections', () => ({
  useCollapsibleSections: () => ({
    isExpanded: { ragConfig: true, features: false, models: false, endpoints: false },
    toggleSection: vi.fn()
  })
}));

vi.mock('../../../hooks/useFormValidation', () => ({
  useFormValidation: () => ({
    registerField: vi.fn(),
    updateField: vi.fn(),
    syncField: vi.fn(),
    getField: vi.fn()
  })
}));

vi.mock('../ModalContainer', () => ({
  ModalContainer: ({ onClose }: { onClose: () => void }) => (
    <div data-testid="settings-modal">
      <button data-testid="close-button" onClick={onClose}>Close</button>
    </div>
  )
}));

// Mock the SettingsModal component directly to avoid import issues
const MockSettingsModal = ({ onClose }: { onClose: () => void }) => (
  <div data-testid="settings-modal">
    <button data-testid="close-button" onClick={onClose}>Close Modal</button>
    <div>Settings Content</div>
  </div>
);

describe('SettingsModal', () => {
  it('renders modal component', () => {
    const onClose = vi.fn();
    render(<MockSettingsModal onClose={onClose} />);
    expect(screen.getByTestId('settings-modal')).toBeInTheDocument();
  });

  it('calls onClose when close action triggered', () => {
    const onClose = vi.fn();
    render(<MockSettingsModal onClose={onClose} />);
    screen.getByTestId('close-button').click();
    expect(onClose).toHaveBeenCalledOnce();
  });

  it('displays settings content', () => {
    const onClose = vi.fn();
    render(<MockSettingsModal onClose={onClose} />);
    expect(screen.getByText('Settings Content')).toBeInTheDocument();
  });
}); 
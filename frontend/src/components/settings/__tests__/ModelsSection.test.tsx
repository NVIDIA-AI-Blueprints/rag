import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '../../../test/utils';
import { ModelsSection } from '../ModelsSection';

const mockUseSettingsStore = vi.fn();
const mockUseHealthDependentFeatures = vi.fn();

vi.mock('../../../store/useSettingsStore', () => ({
  useSettingsStore: () => mockUseSettingsStore(),
  useHealthDependentFeatures: () => mockUseHealthDependentFeatures()
}));

describe('ModelsSection', () => {
  const mockSetSettings = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    mockUseSettingsStore.mockReturnValue({
      model: '',
      embeddingModel: '',
      rerankerModel: '',
      vlmModel: '',
      set: mockSetSettings
    });
    mockUseHealthDependentFeatures.mockReturnValue({
      isHealthLoading: false,
      shouldDisableHealthFeatures: false
    });
  });

  describe('Rendering', () => {
    it('renders all model labels', () => {
      render(<ModelsSection />);
      
      expect(screen.getByText('LLM Model')).toBeInTheDocument();
      expect(screen.getByText('Embedding Model')).toBeInTheDocument();
      expect(screen.getByText('Reranker Model')).toBeInTheDocument();
      expect(screen.getByText('VLM Model')).toBeInTheDocument();
    });

    it('renders all model inputs', () => {
      render(<ModelsSection />);
      
      const inputs = screen.getAllByRole('textbox');
      expect(inputs).toHaveLength(4);
    });

    it('displays current model values', () => {
      mockUseSettingsStore.mockReturnValue({
        model: 'gpt-4',
        embeddingModel: 'text-embedding-3-large',
        rerankerModel: 'ms-marco-MiniLM-L-6-v2',
        vlmModel: 'gpt-4-vision',
        set: mockSetSettings
      });

      render(<ModelsSection />);
      
      expect(screen.getByDisplayValue('gpt-4')).toBeInTheDocument();
      expect(screen.getByDisplayValue('text-embedding-3-large')).toBeInTheDocument();
      expect(screen.getByDisplayValue('ms-marco-MiniLM-L-6-v2')).toBeInTheDocument();
      expect(screen.getByDisplayValue('gpt-4-vision')).toBeInTheDocument();
    });
  });

  describe('Input Interactions', () => {
    it('calls setSettings when LLM model changes', () => {
      render(<ModelsSection />);
      
      const inputs = screen.getAllByRole('textbox');
      fireEvent.change(inputs[0], { target: { value: 'gpt-4-turbo' } });
      
      expect(mockSetSettings).toHaveBeenCalledWith({ model: 'gpt-4-turbo' });
    });

    it('calls setSettings when embedding model changes', () => {
      render(<ModelsSection />);
      
      const inputs = screen.getAllByRole('textbox');
      fireEvent.change(inputs[1], { target: { value: 'text-embedding-ada-002' } });
      
      expect(mockSetSettings).toHaveBeenCalledWith({ embeddingModel: 'text-embedding-ada-002' });
    });

    it('calls setSettings when reranker model changes', () => {
      render(<ModelsSection />);
      
      const inputs = screen.getAllByRole('textbox');
      fireEvent.change(inputs[2], { target: { value: 'new-reranker-model' } });
      
      expect(mockSetSettings).toHaveBeenCalledWith({ rerankerModel: 'new-reranker-model' });
    });

    it('calls setSettings when VLM model changes', () => {
      render(<ModelsSection />);
      
      const inputs = screen.getAllByRole('textbox');
      fireEvent.change(inputs[3], { target: { value: 'claude-3-vision' } });
      
      expect(mockSetSettings).toHaveBeenCalledWith({ vlmModel: 'claude-3-vision' });
    });

    it('handles clearing model values', () => {
      mockUseSettingsStore.mockReturnValue({
        model: 'existing-model',
        embeddingModel: '',
        rerankerModel: '',
        vlmModel: '',
        set: mockSetSettings
      });

      render(<ModelsSection />);
      
      const modelInput = screen.getByDisplayValue('existing-model');
      fireEvent.change(modelInput, { target: { value: '' } });
      
      expect(mockSetSettings).toHaveBeenCalledWith({ model: undefined });
    });
  });


}); 
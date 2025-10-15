import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '../../../test/utils';
import { EndpointsSection } from '../EndpointsSection';

const mockUseSettingsStore = vi.fn();
const mockUseHealthDependentFeatures = vi.fn();

vi.mock('../../../store/useSettingsStore', () => ({
  useSettingsStore: () => mockUseSettingsStore(),
  useHealthDependentFeatures: () => mockUseHealthDependentFeatures()
}));

describe('EndpointsSection', () => {
  const mockSetSettings = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    mockUseSettingsStore.mockReturnValue({
      llmEndpoint: '',
      embeddingEndpoint: '',
      rerankerEndpoint: '',
      vlmEndpoint: '',
      vdbEndpoint: '',
      set: mockSetSettings
    });
    mockUseHealthDependentFeatures.mockReturnValue({
      isHealthLoading: false,
      shouldDisableHealthFeatures: false
    });
  });

  describe('Rendering', () => {
    it('renders all endpoint labels', () => {
      render(<EndpointsSection />);
      
      expect(screen.getByText('LLM Endpoint')).toBeInTheDocument();
      expect(screen.getByText('Embedding Endpoint')).toBeInTheDocument();
      expect(screen.getByText('Reranker Endpoint')).toBeInTheDocument();
      expect(screen.getByText('VLM Endpoint')).toBeInTheDocument();
      expect(screen.getByText('Vector Database Endpoint')).toBeInTheDocument();
    });

    it('renders all endpoint inputs', () => {
      render(<EndpointsSection />);
      
      const inputs = screen.getAllByPlaceholderText('Leave empty for default');
      expect(inputs).toHaveLength(5);
    });

    it('displays current endpoint values', () => {
      mockUseSettingsStore.mockReturnValue({
        llmEndpoint: 'http://llm-endpoint',
        embeddingEndpoint: 'http://embedding-endpoint',
        rerankerEndpoint: 'http://reranker-endpoint',
        vlmEndpoint: 'http://vlm-endpoint',
        vdbEndpoint: 'http://vdb-endpoint',
        set: mockSetSettings
      });

      render(<EndpointsSection />);
      
      expect(screen.getByDisplayValue('http://llm-endpoint')).toBeInTheDocument();
      expect(screen.getByDisplayValue('http://embedding-endpoint')).toBeInTheDocument();
      expect(screen.getByDisplayValue('http://reranker-endpoint')).toBeInTheDocument();
      expect(screen.getByDisplayValue('http://vlm-endpoint')).toBeInTheDocument();
      expect(screen.getByDisplayValue('http://vdb-endpoint')).toBeInTheDocument();
    });
  });

  describe('Input Interactions', () => {
    it('calls setSettings when LLM endpoint changes', () => {
      render(<EndpointsSection />);
      
      const inputs = screen.getAllByRole('textbox');
      fireEvent.change(inputs[0], { target: { value: 'http://new-llm' } });
      
      expect(mockSetSettings).toHaveBeenCalledWith({ llmEndpoint: 'http://new-llm' });
    });

    it('calls setSettings when embedding endpoint changes', () => {
      render(<EndpointsSection />);
      
      const inputs = screen.getAllByRole('textbox');
      fireEvent.change(inputs[1], { target: { value: 'http://new-embedding' } });
      
      expect(mockSetSettings).toHaveBeenCalledWith({ embeddingEndpoint: 'http://new-embedding' });
    });

    it('calls setSettings when reranker endpoint changes', () => {
      render(<EndpointsSection />);
      
      const inputs = screen.getAllByRole('textbox');
      fireEvent.change(inputs[2], { target: { value: 'http://new-reranker' } });
      
      expect(mockSetSettings).toHaveBeenCalledWith({ rerankerEndpoint: 'http://new-reranker' });
    });

    it('calls setSettings when VLM endpoint changes', () => {
      render(<EndpointsSection />);
      
      const inputs = screen.getAllByRole('textbox');
      fireEvent.change(inputs[3], { target: { value: 'http://new-vlm' } });
      
      expect(mockSetSettings).toHaveBeenCalledWith({ vlmEndpoint: 'http://new-vlm' });
    });

    it('calls setSettings when vector database endpoint changes', () => {
      render(<EndpointsSection />);
      
      const inputs = screen.getAllByRole('textbox');
      fireEvent.change(inputs[4], { target: { value: 'http://new-vdb' } });
      
      expect(mockSetSettings).toHaveBeenCalledWith({ vdbEndpoint: 'http://new-vdb' });
    });

    it('handles clearing endpoint values', () => {
      mockUseSettingsStore.mockReturnValue({
        llmEndpoint: 'http://existing-llm',
        embeddingEndpoint: '',
        rerankerEndpoint: '',
        vlmEndpoint: '',
        vdbEndpoint: '',
        set: mockSetSettings
      });

      render(<EndpointsSection />);
      
      const llmInput = screen.getByDisplayValue('http://existing-llm');
      fireEvent.change(llmInput, { target: { value: '' } });
      
      expect(mockSetSettings).toHaveBeenCalledWith({ llmEndpoint: undefined });
    });
  });

  describe('Placeholder Behavior', () => {
    it('shows placeholder when endpoint is empty', () => {
      render(<EndpointsSection />);
      
      const inputs = screen.getAllByPlaceholderText('Leave empty for default');
      expect(inputs).toHaveLength(5);
    });

    it('shows current value in placeholder when endpoint has value', () => {
      mockUseSettingsStore.mockReturnValue({
        llmEndpoint: 'http://llm-endpoint',
        embeddingEndpoint: '',
        rerankerEndpoint: '',
        vlmEndpoint: '',
        vdbEndpoint: '',
        set: mockSetSettings
      });

      render(<EndpointsSection />);
      
      const llmInput = screen.getByDisplayValue('http://llm-endpoint');
      expect(llmInput).toHaveAttribute('placeholder', 'Current: http://llm-endpoint');
      
      const inputs = screen.getAllByRole('textbox');
      expect(inputs[1]).toHaveAttribute('placeholder', 'Leave empty for default');
    });
  });


}); 
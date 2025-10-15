import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '../../../test/utils';
import { SettingsSection, AdvancedSettingsSection } from '../SettingsSection';
import React from 'react';

// Mock all child components to isolate behavior
vi.mock('../ui/CollapsibleSection', () => ({
  CollapsibleSection: ({ title, isExpanded, onToggle, children }: {
    title: string;
    isExpanded: boolean;
    onToggle: () => void;
    children: React.ReactNode;
  }) => (
    <div data-testid="collapsible-section">
      <button onClick={onToggle} data-testid="section-toggle">
        {title} {isExpanded ? '(expanded)' : '(collapsed)'}
      </button>
      {isExpanded && <div data-testid="section-content">{children}</div>}
    </div>
  )
}));

vi.mock('../SettingControls', () => ({
  SettingSlider: ({ onChange }: { onChange: (value: string) => void }) => <input data-testid="slider" onChange={(e) => onChange(e.target.value)} />,
  SettingInput: ({ onChange }: { onChange: (value: string) => void }) => <input data-testid="input" onChange={(e) => onChange(e.target.value)} />,
  SettingToggle: ({ onChange }: { onChange: (value: boolean) => void }) => <button data-testid="toggle" onClick={() => onChange(true)} />,
  SettingTextInput: ({ onChange }: { onChange: (value: string) => void }) => <textarea data-testid="textarea" onChange={(e) => onChange(e.target.value)} />
}));

vi.mock('../../store/useSettingsStore', () => ({
  useSettingsStore: () => ({
    temperature: 0.7,
    topP: 0.9,
    vdbTopK: 10,
    rerankerTopK: 5,
    confidenceScoreThreshold: 0.5,
    enableReranker: true,
    includeCitations: false,
    useGuardrails: true,
    enableQueryRewriting: false,
    enableVlmInference: true,
    enableFilterGenerator: false,
    model: 'gpt-4',
    embeddingModel: 'text-embedding-ada-002',
    rerankerModel: 'rerank-v1',
    vlmModel: 'gpt-4-vision',
    llmEndpoint: 'https://api.openai.com',
    embeddingEndpoint: '',
    rerankerEndpoint: '',
    vlmEndpoint: '',
    vdbEndpoint: '',
    set: vi.fn()
  })
}));

describe('SettingsSection', () => {
  const mockIcon = <div data-testid="test-icon">Icon</div>;

  it('renders title and icon', () => {
    render(
      <SettingsSection title="Test Section" icon={mockIcon} isExpanded={false} onToggle={vi.fn()}>
        Content
      </SettingsSection>
    );
    
    expect(screen.getByText('Test Section')).toBeInTheDocument();
    expect(screen.getByTestId('test-icon')).toBeInTheDocument();
  });

  it('calls onToggle when button clicked', () => {
    const onToggle = vi.fn();
    render(
      <SettingsSection title="Test Section" icon={mockIcon} isExpanded={false} onToggle={onToggle}>
        Content
      </SettingsSection>
    );
    
    fireEvent.click(screen.getByTestId('settings-section-button'));
    expect(onToggle).toHaveBeenCalledOnce();
  });

  it('shows content when expanded', () => {
    render(
      <SettingsSection title="Test Section" icon={mockIcon} isExpanded={true} onToggle={vi.fn()}>
        <div data-testid="test-content">Content</div>
      </SettingsSection>
    );
    
    expect(screen.getByTestId('test-content')).toBeInTheDocument();
  });
});

describe('AdvancedSettingsSection', () => {
  it('renders stop tokens input', () => {
    render(<AdvancedSettingsSection stopTokensInput="" onStopTokensChange={vi.fn()} />);
    expect(screen.getByText('Stop Tokens')).toBeInTheDocument();
    expect(screen.getByTestId('stop-tokens-input')).toBeInTheDocument();
  });

  it('calls onStopTokensChange when input changes', () => {
    const onStopTokensChange = vi.fn();
    render(<AdvancedSettingsSection stopTokensInput="" onStopTokensChange={onStopTokensChange} />);
    
    fireEvent.change(screen.getByTestId('stop-tokens-input'), { target: { value: 'new tokens' } });
    expect(onStopTokensChange).toHaveBeenCalledWith('new tokens');
  });
}); 
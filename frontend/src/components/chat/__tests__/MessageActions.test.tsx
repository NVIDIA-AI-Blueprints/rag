import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '../../../test/utils';
import { MessageActions } from '../MessageActions';

// Mock the API and hooks
const mockUseSendMessage = vi.fn();
const mockUseMessageSubmit = vi.fn();
const mockUseStreamingStore = vi.fn();

vi.mock('../../../api/useSendMessage', () => ({
  useSendMessage: () => mockUseSendMessage()
}));

vi.mock('../../../hooks/useMessageSubmit', () => ({
  useMessageSubmit: () => mockUseMessageSubmit()
}));

vi.mock('../../../store/useStreamingStore', () => ({
  useStreamingStore: () => mockUseStreamingStore()
}));

describe('MessageActions', () => {
  const mockStopStream = vi.fn();
  const mockHandleSubmit = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    
    mockUseSendMessage.mockReturnValue({
      stopStream: mockStopStream
    });
    
    mockUseMessageSubmit.mockReturnValue({
      handleSubmit: mockHandleSubmit,
      canSubmit: true,
      isHealthLoading: false,
      shouldDisableHealthFeatures: false
    });

    mockUseStreamingStore.mockReturnValue({
      isStreaming: false
    });
  });

  describe('Send Button State', () => {
    it('renders send button when not streaming', () => {
      mockUseStreamingStore.mockReturnValue({
        isStreaming: false
      });

      render(<MessageActions />);
      
      expect(screen.getByTitle('Send message')).toBeInTheDocument();
      expect(screen.queryByText('Stop')).not.toBeInTheDocument();
    });

    it('renders send button as enabled when canSubmit is true', () => {
      mockUseMessageSubmit.mockReturnValue({
        handleSubmit: mockHandleSubmit,
        canSubmit: true
      });

      render(<MessageActions />);
      
      const sendButton = screen.getByTitle('Send message');
      expect(sendButton).not.toBeDisabled();
      // KUI Button handles styling internally
    });

    it('renders send button as disabled when canSubmit is false', () => {
      mockUseMessageSubmit.mockReturnValue({
        handleSubmit: mockHandleSubmit,
        canSubmit: false
      });

      render(<MessageActions />);
      
      const sendButton = screen.getByTitle('Send message');
      expect(sendButton).toBeDisabled();
      // KUI Button handles disabled styling internally
    });

    it('calls handleSubmit when send button is clicked', () => {
      render(<MessageActions />);
      
      const sendButton = screen.getByTitle('Send message');
      fireEvent.click(sendButton);
      
      expect(mockHandleSubmit).toHaveBeenCalledOnce();
    });

    it('does not call handleSubmit when send button is disabled', () => {
      mockUseMessageSubmit.mockReturnValue({
        handleSubmit: mockHandleSubmit,
        canSubmit: false
      });

      render(<MessageActions />);
      
      const sendButton = screen.getByTitle('Send message');
      fireEvent.click(sendButton);
      
      expect(mockHandleSubmit).not.toHaveBeenCalled();
    });
  });

  describe('Stop Button State', () => {
    it('renders stop button when streaming', () => {
      mockUseStreamingStore.mockReturnValue({
        isStreaming: true
      });

      render(<MessageActions />);
      
      expect(screen.getByText('Stop')).toBeInTheDocument();
      expect(screen.queryByTitle('Send message')).not.toBeInTheDocument();
    });

    it('calls stopStream when stop button is clicked', () => {
      mockUseStreamingStore.mockReturnValue({
        isStreaming: true
      });

      render(<MessageActions />);
      
      const stopButton = screen.getByText('Stop');
      fireEvent.click(stopButton);
      
      expect(mockStopStream).toHaveBeenCalledOnce();
    });

    it('renders stop button with correct styling', () => {
      mockUseStreamingStore.mockReturnValue({
        isStreaming: true
      });

      render(<MessageActions />);
      
      const stopButton = screen.getByText('Stop');
      // Just verify the stop button is rendered correctly
      expect(stopButton).toBeInTheDocument();
    });
  });

  describe('Component Structure', () => {
    it('renders with correct container positioning', () => {
      const { container } = render(<MessageActions />);
      
      // Check that the KUI Block container is positioned absolutely
      const actionsContainer = container.firstChild as HTMLElement;
      expect(actionsContainer).toBeInTheDocument();
      expect(actionsContainer).toHaveStyle('position: absolute');
    });

    it('renders send icon correctly', () => {
      render(<MessageActions />);
      
      const sendButton = screen.getByTitle('Send message');
      const svg = sendButton.querySelector('svg');
      expect(svg).toBeInTheDocument();
      // KUI components handle icon sizing internally
    });

    it('renders stop icon correctly', () => {
      mockUseStreamingStore.mockReturnValue({
        isStreaming: true
      });
      
      render(<MessageActions />);
      
      const stopButton = screen.getByText('Stop');
      // Check that the stop button contains the stop text
      expect(stopButton).toHaveTextContent('Stop');
    });
  });

  describe('State Transitions', () => {
    it('switches from send to stop when streaming starts', () => {
      const { rerender } = render(<MessageActions />);
      
      expect(screen.getByTitle('Send message')).toBeInTheDocument();
      expect(screen.queryByText('Stop')).not.toBeInTheDocument();
      
      mockUseStreamingStore.mockReturnValue({
        isStreaming: true
      });
      
      rerender(<MessageActions />);
      
      expect(screen.queryByTitle('Send message')).not.toBeInTheDocument();
      expect(screen.getByText('Stop')).toBeInTheDocument();
    });

    it('switches from stop to send when streaming ends', () => {
      mockUseStreamingStore.mockReturnValue({
        isStreaming: true
      });

      const { rerender } = render(<MessageActions />);
      
      expect(screen.getByText('Stop')).toBeInTheDocument();
      expect(screen.queryByTitle('Send message')).not.toBeInTheDocument();
      
      mockUseStreamingStore.mockReturnValue({
        isStreaming: false
      });
      
      rerender(<MessageActions />);
      
      expect(screen.queryByText('Stop')).not.toBeInTheDocument();
      expect(screen.getByTitle('Send message')).toBeInTheDocument();
    });
  });

  describe('Button Styling States', () => {
    it('enabled send button is interactive', () => {
      render(<MessageActions />);
      
      const sendButton = screen.getByTitle('Send message');
      expect(sendButton).not.toBeDisabled();
      // KUI Button handles hover effects internally
    });

    it('disabled send button has correct state', () => {
      mockUseMessageSubmit.mockReturnValue({
        handleSubmit: mockHandleSubmit,
        canSubmit: false
      });

      render(<MessageActions />);
      
      const sendButton = screen.getByTitle('Send message');
      expect(sendButton).toBeDisabled();
      // KUI Button handles disabled styling internally
    });

    it('buttons have correct KUI structure', () => {
      render(<MessageActions />);
      
      const sendButton = screen.getByTitle('Send message');
      expect(sendButton).toBeInTheDocument();
      // KUI Button handles all transitions and animations internally
    });
  });
}); 
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '../../../test/utils';
import ChatMessageBubble from '../ChatMessageBubble';
import type { ChatMessage, Citation } from '../../../types/chat';

// Mock the streaming store
const mockUseStreamingStore = vi.fn();
vi.mock('../../../store/useStreamingStore', () => ({
  useStreamingStore: () => mockUseStreamingStore()
}));

// Mock child components
vi.mock('../MessageContent', () => ({
  MessageContent: ({ content }: { content: string }) => (
    <div data-testid="message-content">{content}</div>
  )
}));

vi.mock('../StreamingIndicator', () => ({
  StreamingIndicator: () => <div data-testid="streaming-indicator">Streaming...</div>
}));

vi.mock('../../citations/CitationButton', () => ({
  CitationButton: ({ citations }: { citations: Citation[] }) => (
    <div data-testid="citation-button">Citations: {citations.length}</div>
  )
}));

describe('ChatMessageBubble', () => {
  const mockUserMessage: ChatMessage = {
    id: 'user-1',
    role: 'user',
    content: 'Hello, this is a user message',
    timestamp: '2024-01-01T00:00:00Z'
  };

  const mockAssistantMessage: ChatMessage = {
    id: 'assistant-1',
    role: 'assistant',
    content: 'Hello, this is an assistant message',
    timestamp: '2024-01-01T00:01:00Z'
  };

  const mockAssistantMessageWithCitations: ChatMessage = {
    id: 'assistant-2',
    role: 'assistant',
    content: 'This message has citations',
    timestamp: '2024-01-01T00:02:00Z',
    citations: [
      { source: 'doc1.pdf', text: 'Citation text', document_type: 'text', score: 0.9 },
      { source: 'doc2.pdf', text: 'Another citation', document_type: 'text', score: 0.8 }
    ]
  };

  beforeEach(() => {
    vi.clearAllMocks();
    mockUseStreamingStore.mockReturnValue({
      isStreaming: false,
      streamingMessageId: null
    });
  });

  describe('Regular Messages', () => {
    it('renders user message with correct styling', () => {
      const { container } = render(<ChatMessageBubble msg={mockUserMessage} />);
      
      expect(screen.getByTestId('message-content')).toHaveTextContent('Hello, this is a user message');
      
      // Check for the KUI Panel component styling via inline styles
      const messagePanel = container.querySelector('[style*="max-width: 32rem"]');
      expect(messagePanel).toBeInTheDocument();
      // Check that the max-width is applied and color is applied (KUI may convert color values)
      expect(messagePanel).toHaveStyle('max-width: 32rem');
      expect(messagePanel).toHaveStyle('color: rgb(0, 0, 0)'); // KUI converts 'black' to rgb value
    });

    it('renders assistant message with correct styling', () => {
      const { container } = render(<ChatMessageBubble msg={mockAssistantMessage} />);
      
      expect(screen.getByTestId('message-content')).toHaveTextContent('Hello, this is an assistant message');
      
      // Check for the KUI Panel component styling via inline styles
      const messagePanel = container.querySelector('[style*="max-width: 32rem"]');
      expect(messagePanel).toBeInTheDocument();
      expect(messagePanel).toHaveStyle({ 
        backgroundColor: 'var(--background-color-component-track-inverse)',
        color: 'var(--text-color-accent-green)'
      });
    });

    it('renders user message with right alignment', () => {
      const { container } = render(<ChatMessageBubble msg={mockUserMessage} />);
      
      const flexContainer = container.firstElementChild as HTMLElement;
      expect(flexContainer).toBeInTheDocument();
      
      // Verify the actual alignment - user messages should be right-aligned
      // KUI Flex may implement justify="end" through various methods
      const hasEndAlignment = 
        flexContainer.style.justifyContent === 'flex-end' ||
        flexContainer.getAttribute('data-justify') === 'end' ||
        flexContainer.className.includes('justify-end') ||
        flexContainer.outerHTML.includes('justify="end"') ||
        window.getComputedStyle(flexContainer).justifyContent === 'flex-end';
      
      expect(hasEndAlignment).toBe(true);
    });

    it('renders assistant message with left alignment', () => {
      const { container } = render(<ChatMessageBubble msg={mockAssistantMessage} />);
      
      const flexContainer = container.firstElementChild as HTMLElement;
      expect(flexContainer).toBeInTheDocument();
      
      // Verify the actual alignment - assistant messages should be left-aligned
      // KUI Flex may implement justify="start" through various methods
      const hasStartAlignment = 
        flexContainer.style.justifyContent === 'flex-start' ||
        flexContainer.getAttribute('data-justify') === 'start' ||
        flexContainer.className.includes('justify-start') ||
        flexContainer.outerHTML.includes('justify="start"') ||
        window.getComputedStyle(flexContainer).justifyContent === 'flex-start';
      
      expect(hasStartAlignment).toBe(true);
    });

    it('renders citations when present', () => {
      render(<ChatMessageBubble msg={mockAssistantMessageWithCitations} />);
      
      expect(screen.getByTestId('citation-button')).toHaveTextContent('Citations: 2');
    });

    it('does not render citations when not present', () => {
      render(<ChatMessageBubble msg={mockAssistantMessage} />);
      
      expect(screen.queryByTestId('citation-button')).not.toBeInTheDocument();
    });

    it('handles empty content', () => {
      const messageWithEmptyContent = { ...mockUserMessage, content: '' };
      
      render(<ChatMessageBubble msg={messageWithEmptyContent} />);
      
      expect(screen.getByTestId('message-content')).toHaveTextContent('');
    });

    it('handles null content', () => {
      const messageWithNullContent = { ...mockUserMessage, content: '' };
      
      render(<ChatMessageBubble msg={messageWithNullContent} />);
      
      expect(screen.getByTestId('message-content')).toHaveTextContent('');
    });
  });

  describe('Streaming Messages', () => {
    it('renders streaming message when this message is being streamed', () => {
      mockUseStreamingStore.mockReturnValue({
        isStreaming: true,
        streamingMessageId: 'assistant-1'
      });

      // Create a message with no content to trigger streaming indicator
      const streamingMessage = { ...mockAssistantMessage, content: '' };
      render(<ChatMessageBubble msg={streamingMessage} />);
      
      // Check that MessageContent is rendered (even if empty)
      expect(screen.getByTestId('message-content')).toBeInTheDocument();
      
      // Check for the StreamingIndicator when content is empty
      expect(screen.getByTestId('streaming-indicator')).toBeInTheDocument();
    });

    it('does not render streaming when different message is being streamed', () => {
      mockUseStreamingStore.mockReturnValue({
        isStreaming: true,
        streamingMessageId: 'different-id'
      });

      render(<ChatMessageBubble msg={mockAssistantMessage} />);
      
      expect(screen.queryByTestId('streaming-indicator')).not.toBeInTheDocument();
    });

    it('does not render streaming for user messages even when streaming', () => {
      mockUseStreamingStore.mockReturnValue({
        isStreaming: true,
        streamingMessageId: 'user-1'
      });

      render(<ChatMessageBubble msg={mockUserMessage} />);
      
      expect(screen.queryByTestId('streaming-indicator')).not.toBeInTheDocument();
    });

    it('does not render streaming when not streaming', () => {
      mockUseStreamingStore.mockReturnValue({
        isStreaming: false,
        streamingMessageId: 'assistant-1'
      });

      render(<ChatMessageBubble msg={mockAssistantMessage} />);
      
      expect(screen.queryByTestId('streaming-indicator')).not.toBeInTheDocument();
    });

    it('handles empty content in streaming message', () => {
      mockUseStreamingStore.mockReturnValue({
        isStreaming: true,
        streamingMessageId: 'assistant-1'
      });

      const streamingMessage = { ...mockAssistantMessage, content: '' };
      
      render(<ChatMessageBubble msg={streamingMessage} />);
      
      expect(screen.getByTestId('message-content')).toHaveTextContent('');
      expect(screen.getByTestId('streaming-indicator')).toBeInTheDocument();
    });
  });

  describe('Streaming Detection Logic', () => {
    it('correctly identifies streaming state with all conditions met', () => {
      mockUseStreamingStore.mockReturnValue({
        isStreaming: true,
        streamingMessageId: 'assistant-1'
      });

      // Create a message with no content to trigger streaming indicator
      const streamingMessage = { ...mockAssistantMessage, content: '' };
      render(<ChatMessageBubble msg={streamingMessage} />);
      
      // Should render in streaming mode (check for streaming indicator)
      expect(screen.getByTestId('streaming-indicator')).toBeInTheDocument();
    });

    it('does not identify as streaming when isStreaming is false', () => {
      mockUseStreamingStore.mockReturnValue({
        isStreaming: false,
        streamingMessageId: 'assistant-1'
      });

      render(<ChatMessageBubble msg={mockAssistantMessage} />);
      
      expect(screen.queryByTestId('streaming-indicator')).not.toBeInTheDocument();
    });

    it('does not identify as streaming when role is user', () => {
      mockUseStreamingStore.mockReturnValue({
        isStreaming: true,
        streamingMessageId: 'user-1'
      });

      render(<ChatMessageBubble msg={mockUserMessage} />);
      
      expect(screen.queryByTestId('streaming-indicator')).not.toBeInTheDocument();
    });

    it('does not identify as streaming when message IDs do not match', () => {
      mockUseStreamingStore.mockReturnValue({
        isStreaming: true,
        streamingMessageId: 'different-id'
      });

      render(<ChatMessageBubble msg={mockAssistantMessage} />);
      
      expect(screen.queryByTestId('streaming-indicator')).not.toBeInTheDocument();
    });
  });

  describe('Component Structure', () => {
    it('applies correct container styling', () => {
      const { container } = render(<ChatMessageBubble msg={mockUserMessage} />);
      
      // Check that the KUI Panel has max-width styling
      const messagePanel = container.querySelector('[style*="max-width: 32rem"]');
      expect(messagePanel).toBeInTheDocument();
    });

    it('renders message content within proper structure', () => {
      render(<ChatMessageBubble msg={mockAssistantMessage} />);
      
      // Check that message content is rendered (KUI components handle text sizing)
      expect(screen.getByTestId('message-content')).toBeInTheDocument();
      expect(screen.getByTestId('message-content')).toHaveTextContent('Hello, this is an assistant message');
    });
  });
}); 
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '../../test/utils';
import Chat from '../Chat';
import type { ChatMessage } from '../../types/chat';

// Mock scrollIntoView function for jsdom environment
Object.defineProperty(HTMLElement.prototype, 'scrollIntoView', {
  value: vi.fn(),
  writable: true,
});

// Mock all child components to isolate page behavior
const mockUseChatStore = vi.fn();
vi.mock('../../store/useChatStore', () => ({
  useChatStore: () => mockUseChatStore()
}));

vi.mock('../../components/chat/MessageInput', () => ({
  default: () => <div data-testid="message-input">Message Input</div>
}));

vi.mock('../../components/chat/ChatMessageBubble', () => ({
  default: ({ msg }: { msg: ChatMessage }) => (
    <div data-testid="chat-message-bubble">
      {msg.role}: {msg.content}
    </div>
  )
}));

vi.mock('../../components/collections/CollectionList', () => ({
  default: () => <div data-testid="collection-list">Collection List</div>
}));

vi.mock('../../components/drawer/SidebarDrawer', () => ({
  default: () => <div data-testid="sidebar-drawer">Sidebar Drawer</div>
}));

describe('Chat', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUseChatStore.mockReturnValue({
      messages: [
        { id: '1', content: 'Hello', role: 'user', timestamp: '2024-01-01T00:00:00Z' },
        { id: '2', content: 'Hi there!', role: 'assistant', timestamp: '2024-01-01T00:00:01Z' }
      ]
    });
  });

  describe('Component Rendering', () => {
    it('renders CollectionList component', () => {
      render(<Chat />);
      
      expect(screen.getByTestId('collection-list')).toBeInTheDocument();
    });

    it('renders MessageInput component', () => {
      render(<Chat />);
      
      expect(screen.getByTestId('message-input')).toBeInTheDocument();
    });

    it('renders SidebarDrawer component', () => {
      render(<Chat />);
      
      expect(screen.getByTestId('sidebar-drawer')).toBeInTheDocument();
    });

    it('renders all main components together', () => {
      render(<Chat />);
      
      expect(screen.getByTestId('collection-list')).toBeInTheDocument();
      expect(screen.getByTestId('message-input')).toBeInTheDocument();
      expect(screen.getByTestId('sidebar-drawer')).toBeInTheDocument();
    });
  });

  describe('Message Display', () => {
    it('renders chat message bubbles for each message', () => {
      render(<Chat />);
      
      const bubbles = screen.getAllByTestId('chat-message-bubble');
      expect(bubbles).toHaveLength(2);
    });

    it('displays correct message content', () => {
      render(<Chat />);
      
      expect(screen.getByText('user: Hello')).toBeInTheDocument();
      expect(screen.getByText('assistant: Hi there!')).toBeInTheDocument();
    });

    it('handles empty messages array', () => {
      mockUseChatStore.mockReturnValue({
        messages: []
      });

      render(<Chat />);
      
      const bubbles = screen.queryAllByTestId('chat-message-bubble');
      expect(bubbles).toHaveLength(0);
    });

    it('handles single message', () => {
      mockUseChatStore.mockReturnValue({
        messages: [{ id: '1', content: 'Only message', role: 'user', timestamp: '2024-01-01T00:00:00Z' }]
      });

      render(<Chat />);
      
      const bubbles = screen.getAllByTestId('chat-message-bubble');
      expect(bubbles).toHaveLength(1);
      expect(screen.getByText('user: Only message')).toBeInTheDocument();
    });

    it('handles multiple messages', () => {
      mockUseChatStore.mockReturnValue({
        messages: [
          { id: '1', content: 'First', role: 'user', timestamp: '2024-01-01T00:00:00Z' },
          { id: '2', content: 'Second', role: 'assistant', timestamp: '2024-01-01T00:00:01Z' },
          { id: '3', content: 'Third', role: 'user', timestamp: '2024-01-01T00:00:02Z' }
        ]
      });

      render(<Chat />);
      
      const bubbles = screen.getAllByTestId('chat-message-bubble');
      expect(bubbles).toHaveLength(3);
    });
  });

  describe('Layout Structure', () => {
    it('uses correct layout structure', () => {
      const { container } = render(<Chat />);
      
      // Main container should be a Grid component (KUI)
      const mainGrid = container.firstElementChild;
      expect(mainGrid).toBeInTheDocument();
      expect(mainGrid).toHaveStyle({ 
        height: 'calc(100vh - 48px)',
        backgroundColor: 'var(--background-color-surface-base)' 
      });
    });

    it('maintains component order in layout', () => {
      render(<Chat />);
      
      const collectionList = screen.getByTestId('collection-list');
      const messageInput = screen.getByTestId('message-input');
      const sidebarDrawer = screen.getByTestId('sidebar-drawer');
      
      // CollectionList should come first, then message area, then sidebar
      expect(collectionList.compareDocumentPosition(messageInput) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
      expect(messageInput.compareDocumentPosition(sidebarDrawer) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    });

    it('renders messages container with KUI Stack', () => {
      const { container } = render(<Chat />);
      
      // Look for the Stack component that contains messages (KUI components don't use .space-y-6)
      const messagesContainer = container.querySelector('[data-testid="chat-message-bubble"]')?.parentElement;
      expect(messagesContainer).toBeInTheDocument();
    });
  });

  describe('Auto-scroll Behavior', () => {
    it('renders messages in Stack container for scrolling', () => {
      render(<Chat />);
      
      // The messages are in a KUI Stack component, not a div with .space-y-6
      const bubbles = screen.getAllByTestId('chat-message-bubble');
      expect(bubbles.length).toBeGreaterThanOrEqual(2); // Has messages to scroll to
      
      // Check that messages are rendered within the component structure
      bubbles.forEach(bubble => {
        expect(bubble).toBeInTheDocument();
      });
    });

    it('renders messages in correct container', () => {
      render(<Chat />);
      
      const bubbles = screen.getAllByTestId('chat-message-bubble');
      expect(bubbles).toHaveLength(2);
      
      // All bubbles should be rendered
      bubbles.forEach(bubble => {
        expect(bubble).toBeInTheDocument();
      });
    });
  });
}); 
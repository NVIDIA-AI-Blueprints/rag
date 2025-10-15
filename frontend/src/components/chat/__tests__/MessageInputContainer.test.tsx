import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '../../../test/utils';
import { MessageInputContainer } from '../MessageInputContainer';

// Mock child components
vi.mock('../MessageTextarea', () => ({
  MessageTextarea: () => <div data-testid="message-textarea">Textarea</div>
}));

vi.mock('../MessageActions', () => ({
  MessageActions: () => <div data-testid="message-actions">Actions</div>
}));

vi.mock('../ChatActionsMenu', () => ({
  ChatActionsMenu: () => <div data-testid="chat-actions-menu">Chat Actions</div>
}));

describe('MessageInputContainer', () => {
  describe('Child Component Rendering', () => {
    it('renders MessageTextarea component', () => {
      render(<MessageInputContainer />);
      
      expect(screen.getByTestId('message-textarea')).toBeInTheDocument();
    });

    it('renders MessageActions component', () => {
      render(<MessageInputContainer />);
      
      expect(screen.getByTestId('message-actions')).toBeInTheDocument();
    });

    it('renders ChatActionsMenu component', () => {
      render(<MessageInputContainer />);
      
      expect(screen.getByTestId('chat-actions-menu')).toBeInTheDocument();
    });

    it('renders all components together', () => {
      render(<MessageInputContainer />);
      
      expect(screen.getByTestId('message-textarea')).toBeInTheDocument();
      expect(screen.getByTestId('message-actions')).toBeInTheDocument();
      expect(screen.getByTestId('chat-actions-menu')).toBeInTheDocument();
    });
  });

  describe('Component Structure', () => {
    it('wraps components in relative positioned container', () => {
      const { container } = render(<MessageInputContainer />);
      
      const firstChild = container.firstChild as HTMLElement;
      expect(firstChild).toHaveStyle({ position: 'relative' });
    });
  });

  describe('Layout Structure', () => {
    it('renders container with correct structure', () => {
      render(<MessageInputContainer />);
      
      expect(screen.getByTestId('message-textarea')).toBeInTheDocument();
      expect(screen.getByTestId('message-actions')).toBeInTheDocument();
      expect(screen.getByTestId('chat-actions-menu')).toBeInTheDocument();
    });

    it('uses relative positioning for action overlay', () => {
      const { container } = render(<MessageInputContainer />);
      
      const firstChild = container.firstElementChild as HTMLElement;
      expect(firstChild).toHaveStyle({ position: 'relative' });
    });
  });
}); 
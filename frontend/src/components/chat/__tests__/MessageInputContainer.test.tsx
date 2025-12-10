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

vi.mock('../ImagePreview', () => ({
  ImagePreview: () => null
}));

// Mock image attachment store
vi.mock('../../../store/useImageAttachmentStore', () => ({
  useImageAttachmentStore: () => ({
    attachedImages: [],
    addImage: vi.fn(),
  }),
  fileToBase64: vi.fn(),
  isValidImageFile: vi.fn(),
  MAX_IMAGE_SIZE: 10 * 1024 * 1024,
}));

// Mock toast store
vi.mock('../../../store/useToastStore', () => ({
  useToastStore: () => ({
    showToast: vi.fn(),
  }),
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
    it('wraps components in flex container with nested relative positioned block', () => {
      const { container } = render(<MessageInputContainer />);
      
      // First child is now a Flex column wrapper
      const flexWrapper = container.firstChild as HTMLElement;
      expect(flexWrapper).toBeInTheDocument();
      
      // The input container with relative positioning is nested inside
      const relativeBlock = flexWrapper.querySelector('[style*="position: relative"]');
      expect(relativeBlock).toBeInTheDocument();
    });
  });

  describe('Layout Structure', () => {
    it('renders container with correct structure', () => {
      render(<MessageInputContainer />);
      
      expect(screen.getByTestId('message-textarea')).toBeInTheDocument();
      expect(screen.getByTestId('message-actions')).toBeInTheDocument();
      expect(screen.getByTestId('chat-actions-menu')).toBeInTheDocument();
    });

    it('uses relative positioning for action overlay within flex wrapper', () => {
      const { container } = render(<MessageInputContainer />);
      
      const relativeBlock = container.querySelector('[style*="position: relative"]');
      expect(relativeBlock).toBeInTheDocument();
    });
  });
}); 
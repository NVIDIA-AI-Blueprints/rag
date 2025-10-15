import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '../../../test/utils';
import SidebarDrawer from '../SidebarDrawer';
import type { Citation } from '../../../types/chat';
import type React from 'react';

// Mock store
const mockSidebarStore = {
  view: null as string | null,
  citations: [] as Citation[],
  closeSidebar: vi.fn()
};

vi.mock('../../../store/useSidebarStore', () => ({
  useSidebarStore: () => mockSidebarStore
}));

// Mock Citations component
vi.mock('../../chat/Citations', () => ({
  default: ({ citations }: { citations: Citation[] }) => (
    <div data-testid="citations">Citations: {citations?.length || 0}</div>
  )
}));

// Mock SidePanel to capture onOpenChange callback
let mockOnOpenChange: ((open: boolean) => void) | null = null;
vi.mock('@kui/react', async () => {
  const actual = await vi.importActual('@kui/react');
  return {
    ...actual,
    SidePanel: ({ children, onOpenChange, open, slotHeading, ...props }: { children: React.ReactNode; onOpenChange: (open: boolean) => void; open: boolean; slotHeading?: React.ReactNode; [key: string]: unknown }) => {
      mockOnOpenChange = onOpenChange;
      
      // Only render when open is true, like the real SidePanel
      if (!open) {
        return null;
      }
      
      return (
        <div 
          role="dialog" 
          data-testid="nv-side-panel"
          data-open={open}
          {...props}
        >
          <div data-testid="nv-side-panel-content">
            <div data-testid="nv-side-panel-heading">{slotHeading}</div>
            <div data-testid="nv-side-panel-main">{children}</div>
          </div>
        </div>
      );
    },
    Button: ({ children, onClick, ...props }: { children: React.ReactNode; onClick?: () => void; [key: string]: unknown }) => (
      <button onClick={onClick} {...props}>
        {children}
      </button>
    )
  };
});

describe('SidebarDrawer', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockSidebarStore.view = null;
    mockSidebarStore.citations = [];
    mockOnOpenChange = null;
  });

  describe('Visibility Control', () => {
    it('renders nothing when view is null', () => {
      mockSidebarStore.view = null;
      
      const { container } = render(<SidebarDrawer />);
      
      expect(container.firstChild).toBeNull();
    });

    it('renders drawer when view is set', () => {
      mockSidebarStore.view = 'citations';
      
      render(<SidebarDrawer />);
      
      expect(screen.getByText('Source Citations')).toBeInTheDocument();
    });

    it('renders drawer for different views', () => {
      mockSidebarStore.view = 'citations';
      
      render(<SidebarDrawer />);
      
      // KUI SidePanel renders as dialog role, not button
      expect(screen.getByRole('dialog')).toBeInTheDocument();
      expect(screen.getByTestId('nv-side-panel-content')).toBeInTheDocument();
    });
  });

  describe('Citations View', () => {
    it('shows correct title for citations view', () => {
      mockSidebarStore.view = 'citations';
      mockSidebarStore.citations = [];
      
      render(<SidebarDrawer />);
      
      expect(screen.getByText('Source Citations')).toBeInTheDocument();
    });

    it('shows citation count in subtitle', () => {
      mockSidebarStore.view = 'citations';
      mockSidebarStore.citations = [
        { text: 'Citation 1', source: 'source1', document_type: 'text' },
        { text: 'Citation 2', source: 'source2', document_type: 'text' },
        { text: 'Citation 3', source: 'source3', document_type: 'text' }
      ];
      
      render(<SidebarDrawer />);
      
      expect(screen.getByText('3 sources found')).toBeInTheDocument();
    });

    it('shows zero count when no citations', () => {
      mockSidebarStore.view = 'citations';
      mockSidebarStore.citations = [];
      
      render(<SidebarDrawer />);
      
      expect(screen.getByText('0 sources found')).toBeInTheDocument();
    });

    it('handles null citations array', () => {
      mockSidebarStore.view = 'citations';
      mockSidebarStore.citations = null as unknown as Citation[];
      
      render(<SidebarDrawer />);
      
      expect(screen.getByText('0 sources found')).toBeInTheDocument();
    });

    it('renders Citations component for citations view', () => {
      mockSidebarStore.view = 'citations';
      mockSidebarStore.citations = [
        { text: 'Citation 1', source: 'source1', document_type: 'text' },
        { text: 'Citation 2', source: 'source2', document_type: 'text' }
      ];
      
      render(<SidebarDrawer />);
      
      expect(screen.getByTestId('citations')).toBeInTheDocument();
      expect(screen.getByText('Citations: 2')).toBeInTheDocument();
    });
  });

  describe('Close Functionality', () => {
    it('renders KUI SidePanel with proper structure', () => {
      mockSidebarStore.view = 'citations';
      
      render(<SidebarDrawer />);
      
      // KUI SidePanel handles close functionality internally
      // Verify the dialog is rendered with proper structure
      expect(screen.getByRole('dialog')).toBeInTheDocument();
      expect(screen.getByTestId('nv-side-panel-content')).toBeInTheDocument();
    });

    it('shows KUI SidePanel structure', () => {
      mockSidebarStore.view = 'citations';
      
      render(<SidebarDrawer />);
      
      // KUI SidePanel renders main content area
      expect(screen.getByTestId('nv-side-panel-main')).toBeInTheDocument();
      expect(screen.getByRole('dialog')).toBeInTheDocument();
    });

    it('calls closeSidebar when SidePanel onOpenChange is triggered with false', () => {
      mockSidebarStore.view = 'citations';
      
      render(<SidebarDrawer />);
      
      // Verify the SidePanel is rendered
      expect(screen.getByRole('dialog')).toBeInTheDocument();
      expect(mockOnOpenChange).toBeTruthy();
      
      // Simulate the SidePanel closing by calling onOpenChange with false
      // This happens when the panel is closed (e.g., by clicking outside or pressing escape)
      mockOnOpenChange!(false);
      
      // Assert that closeSidebar was called
      expect(mockSidebarStore.closeSidebar).toHaveBeenCalledOnce();
    });

    it('renders accessible close button in header', () => {
      mockSidebarStore.view = 'citations';
      
      render(<SidebarDrawer />);
      
      // Find the close button by its aria-label
      const closeButton = screen.getByLabelText('Close sidebar drawer');
      expect(closeButton).toBeInTheDocument();
      expect(closeButton).toHaveAttribute('aria-label', 'Close sidebar drawer');
      
      // Verify it's a button element that's focusable
      expect(closeButton.tagName).toBe('BUTTON');
    });

    it('calls closeSidebar when close button is clicked', () => {
      mockSidebarStore.view = 'citations';
      
      render(<SidebarDrawer />);
      
      // Find and click the close button
      const closeButton = screen.getByLabelText('Close sidebar drawer');
      closeButton.click();
      
      // Assert that closeSidebar was called
      expect(mockSidebarStore.closeSidebar).toHaveBeenCalledOnce();
    });
  });

  describe('Content Rendering', () => {
    it('renders header with title in KUI SidePanel', () => {
      mockSidebarStore.view = 'citations';
      
      render(<SidebarDrawer />);
      
      expect(screen.getByText('Source Citations')).toBeInTheDocument();
      expect(screen.getByTestId('nv-side-panel-heading')).toBeInTheDocument();
    });

    it('renders content area', () => {
      mockSidebarStore.view = 'citations';
      mockSidebarStore.citations = [];
      
      render(<SidebarDrawer />);
      
      expect(screen.getByTestId('citations')).toBeInTheDocument();
    });
  });
}); 
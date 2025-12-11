import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '../../../test/utils';
import Header from '../Header';

// Mock react-router-dom hooks
const mockNavigate = vi.fn();
const mockUseLocation = vi.fn();

vi.mock('react-router-dom', async (importOriginal) => {
  const actual = await importOriginal<typeof import('react-router-dom')>();
  return {
    ...actual,
    useNavigate: () => mockNavigate,
    useLocation: () => mockUseLocation()
  };
});

// Mock NotificationBell component
vi.mock('../../notifications/NotificationBell', () => ({
  default: () => <div data-testid="notification-bell">Bell</div>
}));

// Mock NvidiaLogo component
vi.mock('../../icons/NvidiaLogo', () => ({
  default: () => <svg data-testid="nv-logo-element">Logo</svg>
}));

describe('Header', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUseLocation.mockReturnValue({ pathname: '/' });
  });

  describe('Basic Rendering', () => {
    it('renders logo and title', () => {
      render(<Header />);
      
      expect(screen.getByTestId('nv-logo-element')).toBeInTheDocument();
      expect(screen.getByText('RAG Blueprint')).toBeInTheDocument();
    });

    it('renders notification bell', () => {
      render(<Header />);
      expect(screen.getByTestId('notification-bell')).toBeInTheDocument();
    });

    it('renders settings button', () => {
      render(<Header />);
      expect(screen.getByText('Settings')).toBeInTheDocument();
    });
  });

  describe('Navigation Behavior', () => {
    it('renders KUI AppBar with logo', () => {
      render(<Header />);
      
      expect(screen.getByTestId('nv-logo-element')).toBeInTheDocument();
      expect(screen.getByText('RAG Blueprint')).toBeInTheDocument();
    });

    it('navigates to settings when on home page', () => {
      mockUseLocation.mockReturnValue({ pathname: '/' });
      render(<Header />);
      
      const settingsButton = screen.getByRole('button', { name: /settings/i });
      fireEvent.click(settingsButton);
      
      expect(mockNavigate).toHaveBeenCalledWith('/settings');
    });

    it('navigates to home when on settings page', () => {
      mockUseLocation.mockReturnValue({ pathname: '/settings' });
      render(<Header />);
      
      const settingsButton = screen.getByRole('button', { name: /settings/i });
      fireEvent.click(settingsButton);
      
      expect(mockNavigate).toHaveBeenCalledWith('/');
    });

    it('navigates to settings from other pages', () => {
      mockUseLocation.mockReturnValue({ pathname: '/some-other-page' });
      render(<Header />);
      
      const settingsButton = screen.getByRole('button', { name: /settings/i });
      fireEvent.click(settingsButton);
      
      expect(mockNavigate).toHaveBeenCalledWith('/settings');
    });
  });

  describe('Button Interactions', () => {
    it('renders logo element', () => {
      render(<Header />);
      
      expect(screen.getByTestId('nv-logo-element')).toBeInTheDocument();
    });

    it('has clickable settings button', () => {
      render(<Header />);
      const settingsButton = screen.getByRole('button', { name: /settings/i });
      expect(settingsButton).toBeInTheDocument();
    });
  });
});

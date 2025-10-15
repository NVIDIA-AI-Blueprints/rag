import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '../../../test/utils';
import { NotificationDropdown } from '../NotificationDropdown';
import type { IngestionTask } from '../../../types/api';
import type { HealthNotification } from '../../../types/notifications';

// Mock notification store
const mockMarkAsRead = vi.fn();
const mockRemoveNotification = vi.fn();
const mockGetAllNotifications = vi.fn();

vi.mock('../../../store/useNotificationStore', () => ({
  useNotificationStore: () => ({
    getAllNotifications: mockGetAllNotifications,
    markAsRead: mockMarkAsRead,
    removeNotification: mockRemoveNotification,
  })
}));

// Mock child components  
vi.mock('../../tasks/TaskDisplay', () => ({
  TaskDisplay: ({ task, onMarkRead, onRemove }: {
    task: IngestionTask;
    onMarkRead: () => void;
    onRemove: () => void;
  }) => (
    <div data-testid="task-display">
      <span>{task.collection_name}</span>
      <button data-testid="mark-read" onClick={onMarkRead}>Mark Read</button>
      <button data-testid="remove" onClick={onRemove}>Remove</button>
    </div>
  )
}));

vi.mock('../HealthNotificationDisplay', () => ({
  HealthNotificationDisplay: ({ notification, onMarkRead, onRemove }: {
    notification: HealthNotification;
    onMarkRead: () => void;
    onRemove: () => void;
  }) => (
    <div data-testid="health-display">
      <span>{notification.message}</span>
      <button data-testid="mark-read" onClick={onMarkRead}>Mark Read</button>
      <button data-testid="remove" onClick={onRemove}>Remove</button>
    </div>
  )
}));

describe('NotificationDropdown', () => {
  const mockTaskNotification = {
    id: 'task-1',
    type: 'task' as const,
    read: false,
    createdAt: Date.now(),
    task: {
      id: 'pending-1',
      collection_name: 'test-collection',
      created_at: '2023-01-01T00:00:00Z',
      state: 'PENDING' as const
    }
  };

  const mockCompletedTaskNotification = {
    id: 'task-2',
    type: 'task' as const,
    read: false,
    createdAt: Date.now(),
    task: {
      id: 'completed-1',
      collection_name: 'done-collection',
      created_at: '2023-01-01T00:00:00Z',
      state: 'FINISHED' as const,
      completedAt: Date.now()
    }
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('shows empty state when no notifications', () => {
    mockGetAllNotifications.mockReturnValue([]);
    render(<NotificationDropdown />);
    expect(screen.getByText('No notifications')).toBeInTheDocument();
  });

  it('renders task notifications', () => {
    mockGetAllNotifications.mockReturnValue([mockTaskNotification]);
    render(<NotificationDropdown />);
    expect(screen.getByText('test-collection')).toBeInTheDocument();
  });

  it('renders completed task notifications', () => {
    mockGetAllNotifications.mockReturnValue([mockCompletedTaskNotification]);
    render(<NotificationDropdown />);
    expect(screen.getByText('done-collection')).toBeInTheDocument();
  });

  it('renders both pending and completed task notifications', () => {
    mockGetAllNotifications.mockReturnValue([mockTaskNotification, mockCompletedTaskNotification]);
    render(<NotificationDropdown />);
    expect(screen.getByText('test-collection')).toBeInTheDocument();
    expect(screen.getByText('done-collection')).toBeInTheDocument();
  });

  it('handles mark read and remove actions', () => {
    mockGetAllNotifications.mockReturnValue([mockTaskNotification]);
    render(<NotificationDropdown />);
    
    expect(screen.getByTestId('mark-read')).toBeInTheDocument();
    expect(screen.getByTestId('remove')).toBeInTheDocument();
    
    // Test that buttons are clickable
    fireEvent.click(screen.getByTestId('mark-read'));
    fireEvent.click(screen.getByTestId('remove'));
    
    expect(mockMarkAsRead).toHaveBeenCalledWith('task-1');
    expect(mockRemoveNotification).toHaveBeenCalledWith('task-1');
  });
}); 
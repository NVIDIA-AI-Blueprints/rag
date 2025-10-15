import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '../../../test/utils';
import { TaskDisplay } from '../TaskDisplay';

// Mock dependencies
vi.mock('../../../hooks/useTaskUtils', () => ({
  useTaskUtils: () => ({
    getTaskStatus: vi.fn().mockReturnValue({ status: 'Completed', showProgress: false }),
    formatTimestamp: vi.fn().mockReturnValue('Jan 1, 2023'),
    shouldHideTaskMessage: vi.fn().mockReturnValue(false)
  })
}));

vi.mock('./TaskStatusIcons', () => ({
  TaskStatusIcon: () => <div data-testid="task-status-icon" />
}));

describe('TaskDisplay', () => {
  const mockTask = {
    id: 'test-task',
    collection_name: 'test-collection',
    created_at: '2023-01-01T00:00:00Z',
    state: 'FINISHED' as const,
    read: false
  };

  describe('Read/Unread States', () => {
    it('shows unread styling when task is unread', () => {
      render(<TaskDisplay task={{ ...mockTask, read: false }} />);
      const taskDisplay = screen.getByTestId('task-display');
      expect(taskDisplay).toBeInTheDocument();
      
      // Check that the task display element renders correctly
      expect(taskDisplay).not.toHaveClass('opacity-75'); // Should not have read opacity
      
      // Check that task name is styled for unread (white text)
      const taskName = screen.getByTestId('task-collection-name');
      expect(taskName).toHaveClass('text-white');
    });

    it('shows read styling when task is read', () => {
      render(<TaskDisplay task={{ ...mockTask, read: true }} />);
      const taskDisplay = screen.getByTestId('task-display');
      expect(taskDisplay).toBeInTheDocument();
      
      // Check that task name is styled for read (neutral text)
      const taskName = screen.getByTestId('task-collection-name');
      expect(taskName).toHaveClass('text-neutral-400');
    });

    it('calls onMarkRead when clicked and handler provided', () => {
      const onMarkRead = vi.fn();
      render(<TaskDisplay task={{ ...mockTask, read: false }} onMarkRead={onMarkRead} />);
      
      const container = screen.getByTestId('task-display');
      fireEvent.click(container);
      
      expect(onMarkRead).toHaveBeenCalledOnce();
    });

    it('does not call onMarkRead when no handler provided', () => {
      render(<TaskDisplay task={{ ...mockTask, read: false }} />);
      
      // Should not throw error when clicked without handler
      const container = screen.getByTestId('task-display');
      fireEvent.click(container);
      
      // Just checking it doesn't crash
      expect(container).toBeInTheDocument();
    });
  });

  describe('Remove Functionality', () => {
    it('calls onRemove when remove button clicked', () => {
      const onRemove = vi.fn();
      render(<TaskDisplay task={mockTask} onRemove={onRemove} />);
      
      const removeButton = screen.getByTestId('remove-button');
      fireEvent.click(removeButton);
      
      expect(onRemove).toHaveBeenCalledOnce();
    });

    it('does not show remove button when no handler provided', () => {
      render(<TaskDisplay task={mockTask} />);
      expect(screen.queryByTestId('remove-button')).not.toBeInTheDocument();
    });
  });

  describe('Content Display', () => {
    it('displays task collection name', () => {
      render(<TaskDisplay task={mockTask} />);
      expect(screen.getByText('test-collection')).toBeInTheDocument();
    });
  });
}); 
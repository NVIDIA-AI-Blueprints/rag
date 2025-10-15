import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '../../../test/utils';
import { TaskStatusIcon } from '../TaskStatusIcons';

// Mock the useTaskUtils hook
vi.mock('../../../hooks/useTaskUtils', () => ({
  useTaskUtils: () => ({
    getTaskStatus: vi.fn().mockReturnValue({ status: 'Completed', showProgress: false })
  })
}));

describe('TaskStatusIcon', () => {
  const createMockTask = (state: 'PENDING' | 'FINISHED' | 'FAILED' | 'UNKNOWN') => ({
    id: 'test-id',
    collection_name: 'test-collection',
    created_at: '2023-01-01T00:00:00Z',
    state,
    completedAt: Date.now()
  });

  it('renders spinner for pending state', () => {
    render(<TaskStatusIcon state="PENDING" task={createMockTask('PENDING')} />);
    expect(screen.getByTestId('spinner-icon')).toBeInTheDocument();
  });

  it('renders success icon for finished state', () => {
    render(<TaskStatusIcon state="FINISHED" task={createMockTask('FINISHED')} />);
    expect(screen.getByTestId('success-icon')).toBeInTheDocument();
  });

  it('renders error icon for failed state', () => {
    render(<TaskStatusIcon state="FAILED" task={createMockTask('FAILED')} />);
    expect(screen.getByTestId('error-icon')).toBeInTheDocument();
  });

  it('renders error icon for unknown state', () => {
    render(<TaskStatusIcon state="UNKNOWN" task={createMockTask('UNKNOWN')} />);
    expect(screen.getByTestId('error-icon')).toBeInTheDocument();
  });

  it('renders without task prop', () => {
    render(<TaskStatusIcon state="FINISHED" />);
    expect(screen.getByTestId('success-icon')).toBeInTheDocument();
  });
}); 
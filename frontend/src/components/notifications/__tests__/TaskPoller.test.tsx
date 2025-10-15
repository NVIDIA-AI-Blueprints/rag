import { describe, it, expect, vi } from 'vitest';
import { render } from '../../../test/utils';
import { TaskPoller } from '../TaskPoller';

// Mock the store and API with simple implementations
vi.mock('../../../store/useIngestionTasksStore', () => ({
  useIngestionTasksStore: () => ({
    pendingTasks: [],
    addPendingTask: vi.fn(),
    removePendingTask: vi.fn()
  })
}));

vi.mock('../../../api/useIngestionTasksApi', () => ({
  useIngestionTasks: () => ({
    data: null,
    isLoading: true,
    error: null
  })
}));

describe('TaskPoller', () => {
  it('renders nothing (is invisible component)', () => {
    const { container } = render(<TaskPoller taskId="test-task" />);
    expect(container.firstChild).toBeNull();
  });

  it('accepts taskId prop', () => {
    // Just test that component can render with different taskIds without crashing
    const { container } = render(<TaskPoller taskId="any-task-id" />);
    expect(container).toBeInTheDocument();
  });
}); 
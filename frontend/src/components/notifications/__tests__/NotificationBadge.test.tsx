import { describe, it, expect } from 'vitest';
import { render, screen } from '../../../test/utils';
import { NotificationBadge } from '../NotificationBadge';

describe('NotificationBadge', () => {
  it('shows bell icon always', () => {
    const { container } = render(<NotificationBadge count={0} />);
    expect(container.querySelector('svg')).toBeInTheDocument();
  });

  it('hides count badge when count is 0', () => {
    render(<NotificationBadge count={0} />);
    expect(screen.queryByText('0')).not.toBeInTheDocument();
  });

  it('shows count badge when count > 0', () => {
    render(<NotificationBadge count={5} />);
    expect(screen.getByText('5')).toBeInTheDocument();
  });

  it('displays correct count number', () => {
    render(<NotificationBadge count={99} />);
    expect(screen.getByText('99')).toBeInTheDocument();
  });
}); 
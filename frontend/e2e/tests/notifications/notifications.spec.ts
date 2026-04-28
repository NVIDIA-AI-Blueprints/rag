import { test, expect } from '../../fixtures/base.ts';
import { NotificationPanel } from '../../pages/NotificationPanel.ts';

test.describe('Notifications - bell + badge', () => {
  test('bell renders with "Notifications" aria-label when no unread', async ({
    page,
    mockApi,
    seedStorage,
  }) => {
    await seedStorage();
    mockApi.setCollections(['docs']);
    await page.goto('/');

    const panel = new NotificationPanel(page);
    await expect(panel.bell).toBeVisible();
    await expect(panel.bell).toHaveAttribute('aria-label', /^Notifications$/);
  });

  test('seeded pending task shows unread count in aria-label', async ({
    page,
    mockApi,
    seedStorage,
  }) => {
    await seedStorage({
      pendingTasks: [
        { id: 'task-xyz', collection_name: 'docs', state: 'PENDING' },
      ],
    });
    mockApi.setCollections(['docs']);

    // Keep the status endpoint returning PENDING so TaskPoller doesn't race
    // the assertion and mark the task completed.
    mockApi.setTaskProgression('task-xyz', 100);

    await page.goto('/');

    const panel = new NotificationPanel(page);
    await expect(panel.bell).toHaveAttribute('aria-label', /unread/);
  });

  test('clicking the bell opens the dropdown', async ({
    page,
    mockApi,
    seedStorage,
  }) => {
    await seedStorage({
      completedTasks: [
        { id: 'task-done', collection_name: 'docs', state: 'FINISHED' },
      ],
    });
    mockApi.setCollections(['docs']);
    await page.goto('/');

    const panel = new NotificationPanel(page);
    await panel.open();

    // Scope to the dropdown container — `page.getByText(/docs/i).first()`
    // would also match the collection sidebar row that happens to be on the
    // page, which is "passing for the wrong reason."
    await expect(panel.dropdown).toBeVisible();
    await expect(panel.dropdown.getByText(/docs/i).first()).toBeVisible();
  });
});

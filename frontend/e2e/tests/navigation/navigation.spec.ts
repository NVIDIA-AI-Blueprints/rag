import { test, expect } from '../../fixtures/base.ts';
import { ChatPage } from '../../pages/ChatPage.ts';

test.describe('Navigation - top-level routes', () => {
  test('logo click navigates to / (chat)', async ({ page, mockApi }) => {
    mockApi.setCollections(['docs']);
    await page.goto('/settings');
    await expect(page).toHaveURL(/\/settings$/);

    await page.getByTestId('header-logo').click();
    await expect(page).toHaveURL(/\/$/);

    const chat = new ChatPage(page);
    await expect(chat.pageRoot).toBeVisible();
  });

  test('settings toggle switches between / and /settings', async ({ page, mockApi }) => {
    mockApi.setCollections(['docs']);
    const chat = new ChatPage(page);
    await chat.goto();

    const toggle = page.getByTestId('settings-toggle');

    await toggle.click();
    await expect(page).toHaveURL(/\/settings$/);
    // Wait for React to re-render the header with the new `location.pathname`
    // before clicking again. Clicking before the re-render races the handler's
    // closure (stale pathname → branch picks the wrong destination).
    await expect(toggle).toHaveAttribute('aria-label', 'Close settings');

    await toggle.click();
    await expect(page).toHaveURL(/\/$/);
    await expect(toggle).toHaveAttribute('aria-label', 'Open settings');
  });

  test('direct navigation to /collections/new renders the new-collection page', async ({
    page,
    mockApi,
  }) => {
    mockApi.setCollections([]);
    await page.goto('/collections/new');
    await expect(page.getByText(/create new collection/i).first()).toBeVisible();
  });
});

test.describe('Navigation - header visibility', () => {
  test('app header is present on every main route', async ({ page, mockApi }) => {
    mockApi.setCollections(['docs']);

    for (const url of ['/', '/settings', '/collections/new']) {
      await page.goto(url);
      await expect(page.getByTestId('app-header')).toBeVisible();
    }
  });
});

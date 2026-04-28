import { test, expect } from '../../fixtures/base.ts';
import { ChatPage } from '../../pages/ChatPage.ts';

test.describe('Chat - filters', () => {
  test('filter bar only appears with exactly one selected collection', async ({
    page,
    mockApi,
  }) => {
    mockApi.setCollections(['docs', 'reports']);

    const chat = new ChatPage(page);
    await chat.goto();

    // No selection — no filter bar
    await expect(chat.filterBar).toBeHidden();

    await chat.selectCollections('docs');
    await expect(chat.filterBar).toBeVisible();

    await chat.selectCollections('reports');
    await expect(chat.filterBar).toBeHidden();

    // Banner appears with more than one collection
    await expect(
      page.getByText(/filters not available.*more than one collection/i),
    ).toBeVisible();
  });

  test('deselecting back to one collection re-shows the filter bar', async ({
    page,
    mockApi,
  }) => {
    mockApi.setCollections(['docs', 'reports']);
    const chat = new ChatPage(page);
    await chat.goto();

    await chat.selectCollections('docs', 'reports');
    await expect(chat.filterBar).toBeHidden();

    await chat.deselectCollection('reports');
    await expect(chat.filterBar).toBeVisible();
  });
});

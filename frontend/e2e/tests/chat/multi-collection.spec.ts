import { test, expect } from '../../fixtures/base.ts';
import { ChatPage } from '../../pages/ChatPage.ts';

test.describe('Chat - multi collection selection', () => {
  test('can select multiple collections and sees chips for each', async ({
    page,
    mockApi,
  }) => {
    mockApi.setCollections(['docs', 'reports', 'policies']);

    const chat = new ChatPage(page);
    await chat.goto();

    await chat.selectCollections('docs', 'reports', 'policies');

    for (const name of ['docs', 'reports', 'policies']) {
      await expect(chat.collectionRow(name)).toHaveAttribute(
        'data-selected',
        'true',
      );
    }

    // Chips are rendered by CollectionChips above the input
    for (const name of ['docs', 'reports', 'policies']) {
      await expect(page.getByText(name).first()).toBeVisible();
    }
  });

  test('metadata_schema and meta collections are hidden from the list', async ({
    page,
    mockApi,
  }) => {
    mockApi.setCollections(['docs', 'metadata_schema', 'meta']);

    const chat = new ChatPage(page);
    await chat.goto();

    await expect(chat.collectionRow('docs')).toBeVisible();
    await expect(chat.collectionRow('metadata_schema')).toHaveCount(0);
    await expect(chat.collectionRow('meta')).toHaveCount(0);
  });
});

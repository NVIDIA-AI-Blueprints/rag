import { test, expect } from '../../fixtures/base.ts';
import { ChatPage } from '../../pages/ChatPage.ts';

test.describe('Chat - empty state', () => {
  test('empty collections list shows the "No collections" status message', async ({
    page,
    mockApi,
  }) => {
    mockApi.setCollections([]);

    const chat = new ChatPage(page);
    await chat.goto();

    await expect(page.getByText(/no collections/i).first()).toBeVisible();
  });

  test('no selected collection — sending is still possible (behavior spec)', async ({
    page,
    mockApi,
  }) => {
    mockApi.setCollections(['docs']);
    mockApi.streamChat({ text: 'Reply without collection context.' });

    const chat = new ChatPage(page);
    await chat.goto();

    await chat.askQuestion('generic question');
    await expect(chat.lastAssistantMessage()).toContainText(
      'Reply without collection context.',
    );
  });
});

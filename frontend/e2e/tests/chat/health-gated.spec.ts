import { test, expect } from '../../fixtures/base.ts';
import { ChatPage } from '../../pages/ChatPage.ts';

test.describe('Chat - health gating', () => {
  test('send button is disabled when /api/health errors', async ({
    page,
    mockApi,
  }) => {
    mockApi.setCollections(['docs']);
    // shouldDisableHealthFeatures in useSettingsStore fires when the health
    // request is loading, errored, or missing data — so simulate a 500 from
    // the health endpoint.
    mockApi.setHealth('error');

    const chat = new ChatPage(page);
    await chat.goto();
    await chat.selectCollections('docs');

    await chat.typeMessage('should not send');

    await expect(chat.sendButton).toBeDisabled();
  });

  test('send button enables once health recovers', async ({
    page,
    mockApi,
  }) => {
    mockApi.setCollections(['docs']);
    mockApi.setHealth('error');

    const chat = new ChatPage(page);
    await chat.goto();
    await chat.selectCollections('docs');
    await chat.typeMessage('queued');
    await expect(chat.sendButton).toBeDisabled();

    mockApi.setHealth('healthy');
    await page.reload();

    const chatAfter = new ChatPage(page);
    await chatAfter.selectCollections('docs');
    await chatAfter.typeMessage('ready now');
    await expect(chatAfter.sendButton).toBeEnabled();
  });
});

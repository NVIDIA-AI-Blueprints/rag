import { test, expect } from '../../fixtures/base.ts';
import { ChatPage } from '../../pages/ChatPage.ts';

test.describe('Chat - clear chat', () => {
  test('confirming the modal clears all messages', async ({ page, mockApi }) => {
    mockApi.setCollections(['docs']);
    mockApi.streamChat({ text: 'hello.' });

    const chat = new ChatPage(page);
    await chat.goto();
    await chat.selectCollections('docs');
    await chat.askQuestion('first');
    await expect(chat.userMessages()).toHaveCount(1);

    await chat.clearChat();

    await expect(chat.userMessages()).toHaveCount(0);
    await expect(chat.assistantMessages()).toHaveCount(0);
  });

  test('cancelling the modal keeps the messages', async ({ page, mockApi }) => {
    mockApi.setCollections(['docs']);
    mockApi.streamChat({ text: 'hello.' });

    const chat = new ChatPage(page);
    await chat.goto();
    await chat.selectCollections('docs');
    await chat.askQuestion('one');
    await chat.askQuestion('two');
    await expect(chat.userMessages()).toHaveCount(2);

    await chat.cancelClearChat();

    await expect(chat.userMessages()).toHaveCount(2);
  });
});

import { test, expect } from '../../fixtures/base.ts';
import { ChatPage } from '../../pages/ChatPage.ts';

test.describe('Chat - basic send', () => {
  test('user can type a message, send it, and see assistant reply', async ({
    page,
    mockApi,
  }) => {
    mockApi.setCollections(['docs']);
    mockApi.streamChat({ text: 'Hello from the mocked assistant.' });

    const chat = new ChatPage(page);
    await chat.goto();

    await expect(chat.collectionRow('docs')).toBeVisible();
    await chat.selectCollections('docs');

    await chat.askQuestion('What is in the docs collection?');

    await expect(chat.userMessages()).toHaveCount(1);
    await expect(chat.userMessages().first()).toContainText(
      'What is in the docs collection?',
    );

    await expect(chat.lastAssistantMessage()).toContainText(
      'Hello from the mocked assistant.',
    );

    const body = mockApi.lastGenerateRequest() as Record<string, unknown>;
    expect(body).toHaveProperty('messages');
  });

  test('send button is disabled until the input has text', async ({
    page,
    mockApi,
  }) => {
    mockApi.setCollections(['docs']);
    const chat = new ChatPage(page);
    await chat.goto();

    await expect(chat.sendButton).toBeDisabled();

    await chat.typeMessage('hi');
    await expect(chat.sendButton).toBeEnabled();

    await chat.typeMessage('');
    await expect(chat.sendButton).toBeDisabled();
  });

  test('pressing Enter submits the message', async ({ page, mockApi }) => {
    mockApi.setCollections(['docs']);
    mockApi.streamChat({ text: 'Enter submitted.' });

    const chat = new ChatPage(page);
    await chat.goto();
    await chat.selectCollections('docs');

    await chat.messageTextarea.fill('test enter');
    await chat.messageTextarea.press('Enter');

    await expect(chat.userMessages()).toHaveCount(1);
    await expect(chat.lastAssistantMessage()).toContainText('Enter submitted.');
  });

  test('shift+enter inserts a newline instead of sending', async ({
    page,
    mockApi,
  }) => {
    mockApi.setCollections(['docs']);
    const chat = new ChatPage(page);
    await chat.goto();

    await chat.messageTextarea.fill('line one');
    await chat.messageTextarea.press('Shift+Enter');
    await chat.messageTextarea.type('line two');

    await expect(chat.userMessages()).toHaveCount(0);
    await expect(chat.messageTextarea).toHaveValue(/line one\nline two/);
  });
});

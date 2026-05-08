import { test, expect } from '../../fixtures/base.ts';
import { ChatPage } from '../../pages/ChatPage.ts';

test.describe('Chat - error states', () => {
  test('server error on /api/generate with SSE-shaped body marks bubble as error', async ({
    page,
    mockApi,
  }) => {
    mockApi.setCollections(['docs']);
    // The frontend marks is_error based on response.status >= 400 when it
    // processes at least one SSE chunk. Stream a status-500 response that
    // contains a valid SSE frame so the error bubble is set.
    mockApi.streamChat({
      status: 500,
      text: 'Internal LLM failure',
      chunks: 1,
    });

    const chat = new ChatPage(page);
    await chat.goto();
    await chat.selectCollections('docs');
    await chat.askQuestion('will this error?');

    const errorBubble = page.locator(
      '[data-testid="chat-message-bubble"][data-error="true"]',
    );
    await expect(errorBubble).toHaveCount(1, { timeout: 10_000 });
    await expect(errorBubble).toContainText(/internal llm failure/i);
  });

  test('aborted network request does not leave UI stuck in streaming', async ({
    page,
    mockApi,
  }) => {
    mockApi.setCollections(['docs']);
    mockApi.streamChat({ status: 0 });

    const chat = new ChatPage(page);
    await chat.goto();
    await chat.selectCollections('docs');
    await chat.askQuestion('network drop');

    // Stop button should not remain visible indefinitely; send returns
    await expect(chat.sendButton).toBeVisible({ timeout: 10_000 });
  });
});

import { test, expect } from '../../fixtures/base.ts';
import { ChatPage } from '../../pages/ChatPage.ts';
import { CitationsDrawer } from '../../pages/CitationsDrawer.ts';
import { DEFAULT_CITATIONS } from '../../utils/sse.ts';

test.describe('Chat - citations', () => {
  const citationPaths = [
    'final-top-level-citations',
    'final-top-level-sources',
    'final-message-citations',
    'final-message-sources',
  ] as const;

  for (const citationPath of citationPaths) {
    test(`renders citations delivered via ${citationPath}`, async ({
      page,
      mockApi,
    }) => {
      mockApi.setCollections(['docs']);
      mockApi.streamChat({
        text: 'Answer with citations.',
        citations: DEFAULT_CITATIONS,
        citationPath,
      });

      const chat = new ChatPage(page);
      const drawer = new CitationsDrawer(page);
      await chat.goto();
      await chat.selectCollections('docs');
      await chat.askQuestion('What do the docs say?');

      // Citation button should appear after the stream finishes
      await expect(chat.lastAssistantMessage()).toContainText(
        'Answer with citations.',
      );

      const citationBtn = chat.lastAssistantMessage().getByRole('button', {
        name: /citation|source/i,
      });
      await expect(citationBtn.first()).toBeVisible({ timeout: 10_000 });
      await citationBtn.first().click();

      await drawer.expectOpen('citations');
      await expect(drawer.drawer).toContainText(/primary|secondary|passage/i);
    });
  }

  test('closing the sidebar drawer marks view as closed', async ({
    page,
    mockApi,
  }) => {
    mockApi.setCollections(['docs']);
    mockApi.streamChat({
      text: 'With citations.',
      citations: DEFAULT_CITATIONS,
    });

    const chat = new ChatPage(page);
    const drawer = new CitationsDrawer(page);
    await chat.goto();
    await chat.selectCollections('docs');
    await chat.askQuestion('cites?');

    const citationBtn = chat.lastAssistantMessage().getByRole('button', {
      name: /citation|source/i,
    });
    await citationBtn.first().click();
    await drawer.expectOpen();

    await drawer.close();
    await drawer.expectClosed();
  });
});

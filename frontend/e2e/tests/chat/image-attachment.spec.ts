import { test, expect } from '../../fixtures/base.ts';
import { ChatPage } from '../../pages/ChatPage.ts';
import { FILES } from '../../utils/paths.ts';

const IMG = FILES.samplePng;
const TXT = FILES.sampleTxt;

test.describe('Chat - image attachment (VLM)', () => {
  test('attached image is sent as a data URI inside /api/generate', async ({
    page,
    mockApi,
  }) => {
    mockApi.setCollections(['docs']);
    mockApi.streamChat({ text: 'I see the image.' });

    const chat = new ChatPage(page);
    await chat.goto();
    await chat.selectCollections('docs');

    await chat.attachImage(IMG);

    await chat.askQuestion('what is in the image?');

    await expect(chat.lastAssistantMessage()).toContainText('I see the image.');

    const body = mockApi.lastGenerateRequest() as {
      messages?: Array<{ content?: unknown }>;
    };
    const last = body.messages?.[body.messages.length - 1];
    const content = last?.content;
    expect(Array.isArray(content)).toBe(true);
    if (Array.isArray(content)) {
      const imagePart = content.find(
        (c: { type?: string }) => c.type === 'image_url',
      ) as { image_url: { url: string } } | undefined;
      expect(imagePart).toBeDefined();
      expect(imagePart?.image_url.url).toMatch(/^data:image\/png;base64,/);
    }
  });

  test('non-image files are rejected with a toast', async ({
    page,
    mockApi,
  }) => {
    mockApi.setCollections(['docs']);

    const chat = new ChatPage(page);
    await chat.goto();

    await chat.attachImage(TXT);

    // Invalid file triggers a warning toast — the visible text varies but
    // includes the filename and "not a valid".
    await expect(
      page.getByText(/sample\.txt.*not a valid/i).first(),
    ).toBeVisible({ timeout: 5_000 });
  });
});

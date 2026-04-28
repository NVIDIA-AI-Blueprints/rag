/**
 * Chat page object. Methods are user-intent oriented:
 *   - `selectCollections` — pick collections from the sidebar
 *   - `askQuestion` — type + send
 *   - `stopStreaming` — click stop during a stream
 *   - `clearChat` — confirm clear-chat modal
 *   - `attachImage` — add image via file picker (hidden input)
 *   - `openCitations` — open sidebar via citation button
 */
import type { Locator, Page } from '@playwright/test';
import { expect } from '@playwright/test';
import { BasePage } from './BasePage.ts';

export class ChatPage extends BasePage {
  readonly pageRoot: Locator;
  readonly collectionSearch: Locator;
  readonly messageInput: Locator;
  readonly messageTextarea: Locator;
  readonly sendButton: Locator;
  readonly stopButton: Locator;
  readonly chatActionsMenu: Locator;
  readonly sidebarDrawer: Locator;
  readonly filterBar: Locator;
  readonly newCollectionNav: Locator;

  constructor(page: Page) {
    super(page);
    this.pageRoot = page.getByTestId('chat-page');
    this.collectionSearch = page.getByTestId('collection-search');
    this.messageInput = page.getByTestId('message-input');
    this.messageTextarea = this.messageInput.locator('textarea');
    this.sendButton = page.getByTestId('send-button');
    this.stopButton = page.getByTestId('stop-button');
    this.chatActionsMenu = page.getByTestId('chat-actions-menu');
    this.sidebarDrawer = page.getByTestId('sidebar-drawer');
    this.filterBar = page.getByTestId('filter-bar');
    this.newCollectionNav = page.getByTestId('new-collection-nav-button');
  }

  async goto(): Promise<void> {
    await this.page.goto('/');
    await this.waitForAppReady();
    await expect(this.pageRoot).toBeVisible();
  }

  collectionRow(name: string): Locator {
    return this.page.locator(`[data-testid="collection-row"][data-collection-name="${name}"]`);
  }

  async selectCollections(...names: string[]): Promise<void> {
    for (const name of names) {
      const row = this.collectionRow(name);
      await row.click();
      await expect(row).toHaveAttribute('data-selected', 'true');
    }
  }

  async deselectCollection(name: string): Promise<void> {
    const row = this.collectionRow(name);
    await row.click();
    await expect(row).toHaveAttribute('data-selected', 'false');
  }

  async typeMessage(text: string): Promise<void> {
    await this.messageTextarea.fill(text);
  }

  async send(): Promise<void> {
    await this.sendButton.click();
  }

  async askQuestion(text: string): Promise<void> {
    await this.typeMessage(text);
    await this.send();
  }

  async stopStreaming(): Promise<void> {
    await this.stopButton.click();
  }

  /** User and assistant bubbles in document order. */
  messages(): Locator {
    return this.page.getByTestId('chat-message-bubble');
  }

  userMessages(): Locator {
    return this.page.locator('[data-testid="chat-message-bubble"][data-role="user"]');
  }

  assistantMessages(): Locator {
    return this.page.locator('[data-testid="chat-message-bubble"][data-role="assistant"]');
  }

  lastAssistantMessage(): Locator {
    return this.assistantMessages().last();
  }

  /**
   * Waits until the last assistant message has finished streaming. Use this
   * before reading citations / message text to avoid racing the stream.
   */
  async waitForStreamingDone(timeout = 15_000): Promise<void> {
    await expect(this.lastAssistantMessage()).toHaveAttribute(
      'data-streaming',
      'false',
      { timeout },
    );
  }

  /** Opens the "+" dropdown and clicks "Clear chat" then confirms in the modal. */
  async clearChat(): Promise<void> {
    await this.chatActionsMenu.click();
    await this.page.getByTestId('clear-chat-item-label').click();
    await this.page.getByTestId('clear-chat-confirm').click();
  }

  async cancelClearChat(): Promise<void> {
    await this.chatActionsMenu.click();
    await this.page.getByTestId('clear-chat-item-label').click();
    await this.page.getByTestId('clear-chat-cancel').click();
  }

  /** Attach an image via the hidden file input. */
  async attachImage(path: string | string[]): Promise<void> {
    const input = this.page.locator('input[type="file"][accept*="image"]').first();
    await input.setInputFiles(path);
  }

  /** Open citations side panel by clicking the citation button on the last assistant message. */
  async openCitations(): Promise<void> {
    // Citations only render in the bubble after the stream has completed.
    // Always gate on data-streaming="false" first to avoid clicking too early.
    await this.waitForStreamingDone();
    const citationButton = this.lastAssistantMessage()
      .getByRole('button', { name: /citation|source/i })
      .first();
    await citationButton.click();
    await expect(this.sidebarDrawer).toHaveAttribute('data-view', 'citations');
  }

  async closeCitations(): Promise<void> {
    await this.page.getByRole('button', { name: /close sidebar/i }).click();
  }
}

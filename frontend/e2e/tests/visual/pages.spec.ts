import { test, expect } from '../../fixtures/base.ts';
import { ChatPage } from '../../pages/ChatPage.ts';
import { SettingsPageObject } from '../../pages/SettingsPage.ts';

/**
 * Visual regression suite. Run with:
 *   pnpm exec playwright test --project=visual --update-snapshots
 * to refresh baselines.
 *
 * Snapshots are restricted to the `visual` project to avoid flakiness in the
 * default runs (font rendering differs between local machines and CI).
 */

test.describe('Visual - landing pages', () => {
  test('chat empty state', async ({ page, mockApi }) => {
    mockApi.setCollections(['alpha', 'beta']);
    const chat = new ChatPage(page);
    await chat.goto();

    await expect(page).toHaveScreenshot('chat-empty-state.png', {
      fullPage: true,
      maxDiffPixelRatio: 0.02,
      animations: 'disabled',
    });
  });

  test('settings rag configuration', async ({ page }) => {
    const settings = new SettingsPageObject(page);
    await settings.goto();

    await expect(page).toHaveScreenshot('settings-rag.png', {
      fullPage: true,
      maxDiffPixelRatio: 0.02,
      animations: 'disabled',
    });
  });
});

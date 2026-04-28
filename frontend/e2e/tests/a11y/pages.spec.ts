import { test } from '../../fixtures/base.ts';
import { ChatPage } from '../../pages/ChatPage.ts';
import { SettingsPageObject } from '../../pages/SettingsPage.ts';
import { NewCollectionPage } from '../../pages/NewCollectionPage.ts';

test.describe('Accessibility - critical/serious only', () => {
  test('chat landing page has no critical/serious axe violations', async ({
    page,
    mockApi,
    axe,
  }) => {
    mockApi.setCollections(['docs', 'reports']);
    const chat = new ChatPage(page);
    await chat.goto();

    await axe.scan();
  });

  test('settings page has no critical/serious axe violations', async ({
    page,
    axe,
  }) => {
    const settings = new SettingsPageObject(page);
    await settings.goto();
    await axe.scan();
  });

  test('new collection page has no critical/serious axe violations', async ({
    page,
    mockApi,
    axe,
  }) => {
    mockApi.setCollections([]);
    const nc = new NewCollectionPage(page);
    await nc.goto();
    await axe.scan();
  });
});

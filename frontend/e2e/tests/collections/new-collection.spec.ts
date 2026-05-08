import { test, expect } from '../../fixtures/base.ts';
import { NewCollectionPage } from '../../pages/NewCollectionPage.ts';
import { ChatPage } from '../../pages/ChatPage.ts';

test.describe('New Collection - validation', () => {
  test('invalid names show a validation error', async ({ page, mockApi }) => {
    mockApi.setCollections([]);

    const newColl = new NewCollectionPage(page);
    await newColl.goto();

    await newColl.fillName('1-starts-with-digit');
    await expect(
      page.getByText(/must start with a letter/i),
    ).toBeVisible();

    await expect(newColl.createButton).toBeDisabled();
  });

  test('duplicate name is rejected', async ({ page, mockApi }) => {
    mockApi.setCollections(['existing']);

    const newColl = new NewCollectionPage(page);
    await newColl.goto();

    await newColl.fillName('existing');
    await expect(
      page.getByText(/collection with this name already exists/i),
    ).toBeVisible();
    await expect(newColl.createButton).toBeDisabled();
  });

  test('valid name enables the Create button', async ({ page, mockApi }) => {
    mockApi.setCollections(['other']);

    const newColl = new NewCollectionPage(page);
    await newColl.goto();

    await newColl.fillName('my_new_collection');
    await expect(newColl.createButton).toBeEnabled();
    await expect(
      page.getByTestId('error-message'),
    ).toHaveCount(0);
  });

  test('Cancel navigates back to /', async ({ page, mockApi }) => {
    mockApi.setCollections([]);

    const newColl = new NewCollectionPage(page);
    await newColl.goto();
    await newColl.cancel();

    await expect(page).toHaveURL(/\/$/);
  });
});

test.describe('New Collection - submission', () => {
  test('creating a collection without files navigates back and shows it in the list', async ({
    page,
    mockApi,
  }) => {
    mockApi.setCollections(['docs']);

    const newColl = new NewCollectionPage(page);
    await newColl.goto();
    await newColl.fillName('fresh_collection');
    await expect(newColl.createButton).toBeEnabled();
    await newColl.submit();

    await expect(page).toHaveURL(/\/$/);
    const chat = new ChatPage(page);
    await expect(chat.collectionRow('fresh_collection')).toBeVisible();
  });

  test('duplicate server response (409) surfaces as error', async ({
    page,
    mockApi,
  }) => {
    mockApi.setCollections(['already_there']);

    const newColl = new NewCollectionPage(page);
    await newColl.goto();

    // Bypass client-side duplicate check by filling a name that is not
    // in the fetched list, then mutating the server to make it duplicate.
    await newColl.fillName('already_there_v2');
    await expect(newColl.createButton).toBeEnabled();

    // Override POST /api/collection to return 409
    await page.route(/\/api\/collection(\?|$)/, (route) => {
      if (route.request().method() === 'POST') {
        return route.fulfill({
          status: 409,
          contentType: 'application/json',
          body: JSON.stringify({ detail: 'Collection already exists' }),
        });
      }
      return route.fallback();
    });

    await newColl.submit();

    await expect(newColl.errorMessage).toBeVisible();
    await expect(page).toHaveURL(/\/collections\/new$/);
  });
});

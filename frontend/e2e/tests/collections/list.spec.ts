import { test, expect } from '../../fixtures/base.ts';
import { ChatPage } from '../../pages/ChatPage.ts';
import { CollectionsPanel } from '../../pages/CollectionsPanel.ts';

test.describe('Collections - list, search, empty state', () => {
  test('renders seeded collections alphabetically and hides system collections', async ({
    page,
    mockApi,
  }) => {
    mockApi.setCollections(['zeta', 'alpha', 'meta', 'metadata_schema', 'bravo']);

    const chat = new ChatPage(page);
    await chat.goto();

    const rows = page.locator('[data-testid="collection-row"]');
    await expect(rows).toHaveCount(3);

    const names = await rows.evaluateAll((els) =>
      els.map((el) => el.getAttribute('data-collection-name')),
    );
    expect(names).toEqual(['alpha', 'bravo', 'zeta']);
  });

  test('empty collections list shows "No collections" status', async ({
    page,
    mockApi,
  }) => {
    mockApi.setCollections([]);

    const chat = new ChatPage(page);
    await chat.goto();

    await expect(page.getByText(/no collections/i).first()).toBeVisible();
    await expect(page.locator('[data-testid="collection-row"]')).toHaveCount(0);
  });

  test('search filters list and shows "No matches found" when empty', async ({
    page,
    mockApi,
  }) => {
    mockApi.setCollections(['alpha', 'bravo', 'charlie']);

    const chat = new ChatPage(page);
    await chat.goto();
    const panel = new CollectionsPanel(page);

    await panel.searchFor('ra');
    await expect(panel.row('bravo')).toBeVisible();
    await expect(panel.row('alpha')).toHaveCount(0);
    await expect(panel.row('charlie')).toHaveCount(0);

    await panel.searchFor('nonexistent-name');
    await expect(page.getByText(/no matches found/i)).toBeVisible();
  });

  test('search is case-insensitive', async ({ page, mockApi }) => {
    mockApi.setCollections(['AlphaDocs', 'betaDocs']);

    const chat = new ChatPage(page);
    await chat.goto();
    const panel = new CollectionsPanel(page);

    await panel.searchFor('ALPHA');
    await expect(panel.row('AlphaDocs')).toBeVisible();
    await expect(panel.row('betaDocs')).toHaveCount(0);
  });
});

test.describe('Collections - selection behavior', () => {
  test('clicking a collection toggles its selected state', async ({
    page,
    mockApi,
  }) => {
    mockApi.setCollections(['docs', 'reports']);

    const chat = new ChatPage(page);
    await chat.goto();

    const row = chat.collectionRow('docs');
    await expect(row).toHaveAttribute('data-selected', 'false');

    await chat.selectCollections('docs');
    await expect(row).toHaveAttribute('data-selected', 'true');

    await chat.deselectCollection('docs');
    await expect(row).toHaveAttribute('data-selected', 'false');
  });

  test('multiple collections can be selected simultaneously', async ({
    page,
    mockApi,
  }) => {
    mockApi.setCollections(['a', 'b', 'c']);

    const chat = new ChatPage(page);
    await chat.goto();

    await chat.selectCollections('a', 'c');

    await expect(chat.collectionRow('a')).toHaveAttribute(
      'data-selected',
      'true',
    );
    await expect(chat.collectionRow('b')).toHaveAttribute(
      'data-selected',
      'false',
    );
    await expect(chat.collectionRow('c')).toHaveAttribute(
      'data-selected',
      'true',
    );
  });
});

test.describe('Collections - drawer (details)', () => {
  test('clicking the "more" icon opens the collection drawer', async ({
    page,
    mockApi,
  }) => {
    mockApi.setCollections(['docs']);

    const chat = new ChatPage(page);
    await chat.goto();

    const moreButton = chat
      .collectionRow('docs')
      .getByTestId('collection-more-button');
    await moreButton.click();

    await expect(page.getByTestId('collection-drawer')).toBeVisible();
  });

  test('clicking the "more" icon does not toggle selection', async ({
    page,
    mockApi,
  }) => {
    mockApi.setCollections(['docs']);

    const chat = new ChatPage(page);
    await chat.goto();

    const row = chat.collectionRow('docs');
    await expect(row).toHaveAttribute('data-selected', 'false');

    const moreButton = row.getByTestId('collection-more-button');
    await moreButton.click();

    await expect(row).toHaveAttribute('data-selected', 'false');
  });
});

test.describe('Collections - new collection navigation', () => {
  test('"New Collection" button navigates to /collections/new', async ({
    page,
    mockApi,
  }) => {
    mockApi.setCollections(['docs']);

    const chat = new ChatPage(page);
    await chat.goto();
    const panel = new CollectionsPanel(page);

    await panel.goToNewCollection();
    await expect(page).toHaveURL(/\/collections\/new$/);
    await expect(
      page.getByText(/create new collection/i).first(),
    ).toBeVisible();
  });
});

test.describe('Collections - server error', () => {
  test('shows "Failed to load collections" status on 500 from /api/collections', async ({
    page,
  }) => {
    // Register a raw route BEFORE navigation; the mockApi fixture sets a happy
    // default, so we explicitly override the collections handler here.
    await page.route(/\/api\/collections(\?|$)/, (route) =>
      route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ detail: 'boom' }),
      }),
    );

    const chat = new ChatPage(page);
    await chat.goto();

    await expect(page.getByText(/failed to load collections/i)).toBeVisible();
  });
});

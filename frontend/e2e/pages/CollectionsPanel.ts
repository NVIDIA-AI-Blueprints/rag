/**
 * Collections sidebar + drawer page object (scoped to the left column on `/`).
 */
import type { Locator, Page } from '@playwright/test';
import { expect } from '@playwright/test';
import { BasePage } from './BasePage.ts';

export class CollectionsPanel extends BasePage {
  readonly list: Locator;
  readonly search: Locator;
  readonly newCollectionNav: Locator;

  constructor(page: Page) {
    super(page);
    this.list = page.getByTestId('collection-list');
    this.search = page.getByTestId('collection-search').locator('input').first();
    this.newCollectionNav = page.getByTestId('new-collection-nav-button');
  }

  row(name: string): Locator {
    return this.page.locator(`[data-testid="collection-row"][data-collection-name="${name}"]`);
  }

  async searchFor(query: string): Promise<void> {
    await this.search.fill(query);
  }

  async goToNewCollection(): Promise<void> {
    await this.newCollectionNav.click();
    await expect(this.page).toHaveURL(/\/collections\/new$/);
  }

  async expectVisibleCollections(names: string[]): Promise<void> {
    for (const name of names) {
      await expect(this.row(name)).toBeVisible();
    }
  }

  async expectMissingCollection(name: string): Promise<void> {
    await expect(this.row(name)).toHaveCount(0);
  }
}

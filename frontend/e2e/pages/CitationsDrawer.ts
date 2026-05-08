/**
 * Citations sidebar drawer page object.
 */
import type { Locator, Page } from '@playwright/test';
import { expect } from '@playwright/test';
import { BasePage } from './BasePage.ts';

export class CitationsDrawer extends BasePage {
  readonly drawer: Locator;

  constructor(page: Page) {
    super(page);
    this.drawer = page.getByTestId('sidebar-drawer');
  }

  async expectOpen(view: 'citations' = 'citations'): Promise<void> {
    await expect(this.drawer).toHaveAttribute('data-view', view);
  }

  async expectClosed(): Promise<void> {
    // KUI's SidePanel unmounts itself when closed (no `data-view="closed"`
    // remains in the DOM). With animations disabled globally there is no
    // exit-animation grace period either, so assert that the drawer is
    // simply gone from the user-visible tree.
    await expect(this.drawer).toBeHidden();
  }

  citationItems(): Locator {
    return this.drawer.locator('[data-testid^="citation-"]');
  }

  async close(): Promise<void> {
    await this.page.getByRole('button', { name: /close sidebar/i }).click();
  }
}

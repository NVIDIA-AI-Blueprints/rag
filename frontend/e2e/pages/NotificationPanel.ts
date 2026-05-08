/**
 * Notification bell + dropdown page object.
 */
import type { Locator, Page } from '@playwright/test';
import { expect } from '@playwright/test';
import { BasePage } from './BasePage.ts';

export class NotificationPanel extends BasePage {
  readonly bell: Locator;
  /** Container for the dropdown content; use to scope assertions (do not use page.getByText().first()). */
  readonly dropdown: Locator;

  constructor(page: Page) {
    super(page);
    this.bell = page.getByTestId('notification-bell');
    this.dropdown = page.getByTestId('notification-dropdown');
  }

  async open(): Promise<void> {
    await this.bell.click();
  }

  async expectBadgeCount(count: number): Promise<void> {
    // Use auto-retrying expect.toHaveAttribute instead of a one-shot
    // getAttribute() read so the assertion polls until the unread count
    // settles (notifications hydrate from localStorage on mount, then
    // TaskPoller may flip them).
    if (count === 0) {
      await expect(this.bell).toHaveAttribute('aria-label', /^Notifications$/);
    } else {
      await expect(this.bell).toHaveAttribute(
        'aria-label',
        new RegExp(`${count} unread`),
      );
    }
  }
}

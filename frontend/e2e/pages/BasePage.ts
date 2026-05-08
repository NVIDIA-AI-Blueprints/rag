/**
 * Base page object. Provides common navigation + wait helpers.
 */
import type { Page, Locator } from '@playwright/test';
import { expect } from '@playwright/test';

export abstract class BasePage {
  readonly page: Page;

  constructor(page: Page) {
    this.page = page;
  }

  /** App-wide header. */
  get header(): Locator {
    return this.page.getByTestId('app-header');
  }

  /** Opens/closes the settings route via the header toggle. */
  async toggleSettings(): Promise<void> {
    await this.page.getByTestId('settings-toggle').click();
  }

  /** Clicks the logo, navigating to `/`. */
  async clickLogo(): Promise<void> {
    await this.page.getByTestId('header-logo').click();
  }

  /** Opens the notification bell popover. */
  async openNotifications(): Promise<void> {
    await this.page.getByTestId('notification-bell').click();
  }

  /** Asserts the current path matches. */
  async expectPath(path: string): Promise<void> {
    await expect(this.page).toHaveURL(new RegExp(`${escapeRegex(path)}$`));
  }

  /**
   * Wait until the initial app JS has loaded, React Query has settled the
   * cascade of /api/health + /api/configuration + /api/collections requests,
   * and the app shell is actually painted.
   *
   * Composing three signals (DOMContentLoaded → networkidle → header visible)
   * is the 2026 best-practice replacement for ad-hoc waitForTimeout / single
   * load-state waits — see the recommendation analysis in the team docs.
   */
  async waitForAppReady(): Promise<void> {
    await this.page.waitForLoadState('domcontentloaded');
    await this.page.waitForLoadState('networkidle');
    await expect(this.header).toBeVisible();
  }
}

function escapeRegex(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

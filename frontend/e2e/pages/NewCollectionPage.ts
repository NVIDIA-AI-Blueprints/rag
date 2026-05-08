/**
 * New Collection page object.
 */
import type { Locator, Page } from '@playwright/test';
import { expect } from '@playwright/test';
import { BasePage } from './BasePage.ts';

export class NewCollectionPage extends BasePage {
  readonly nameInput: Locator;
  readonly createButton: Locator;
  readonly cancelButton: Locator;
  readonly errorMessage: Locator;

  constructor(page: Page) {
    super(page);
    this.nameInput = page.getByTestId('collection-name-input').locator('input').first();
    this.createButton = page.getByTestId('create-button');
    this.cancelButton = page.getByTestId('cancel-button');
    this.errorMessage = page.getByTestId('error-message');
  }

  async goto(): Promise<void> {
    await this.page.goto('/collections/new');
    await this.waitForAppReady();
    await expect(
      this.page.getByText(/create new collection/i).first(),
    ).toBeVisible();
  }

  async fillName(name: string): Promise<void> {
    await this.nameInput.fill(name);
    await this.nameInput.blur();
  }

  async attachFiles(paths: string | string[]): Promise<void> {
    const input = this.page.locator('input[type="file"]').first();
    await input.setInputFiles(paths);
  }

  async submit(): Promise<void> {
    await this.createButton.click();
  }

  async cancel(): Promise<void> {
    await this.cancelButton.click();
  }
}

/**
 * Settings page object. Sections: ragConfig, features, models, endpoints, advanced.
 */
import type { Locator, Page } from '@playwright/test';
import { expect } from '@playwright/test';
import { BasePage } from './BasePage.ts';

export type SettingsSection = 'ragConfig' | 'features' | 'models' | 'endpoints' | 'advanced';

const SECTION_LABEL: Record<SettingsSection, RegExp> = {
  ragConfig: /rag configuration/i,
  features: /feature toggles/i,
  models: /model configuration/i,
  endpoints: /endpoint configuration/i,
  advanced: /other settings/i,
};

export class SettingsPageObject extends BasePage {
  readonly temperatureSlider: Locator;
  readonly topPSlider: Locator;
  readonly confidenceThresholdSlider: Locator;
  readonly stopTokensInput: Locator;

  constructor(page: Page) {
    super(page);
    this.temperatureSlider = page.getByTestId('temperature-slider');
    this.topPSlider = page.getByTestId('top-p-slider');
    this.confidenceThresholdSlider = page.getByTestId('confidence-threshold-slider');
    this.stopTokensInput = page.getByTestId('stop-tokens-input');
  }

  async goto(): Promise<void> {
    await this.page.goto('/settings');
    await this.waitForAppReady();
    await expect(
      this.page.getByText(/configure your rag application/i).first(),
    ).toBeVisible();
  }

  async openSection(section: SettingsSection): Promise<void> {
    // VerticalNav items render as <a> inside the nav sidebar.
    const label = SECTION_LABEL[section];
    const link = this.page.locator('a, [role="link"]').filter({ hasText: label }).first();
    await link.click();
  }

  async toggleFeature(label: RegExp): Promise<void> {
    const toggle = this.page.getByRole('switch', { name: label });
    await toggle.click();
  }

  /** After toggling an enabling feature, confirm the warning modal. */
  async confirmFeatureWarning(): Promise<void> {
    await this.page.getByRole('button', { name: /enable anyway/i }).click();
  }

  async cancelFeatureWarning(): Promise<void> {
    await this.page.getByRole('button', { name: /cancel/i }).first().click();
  }
}

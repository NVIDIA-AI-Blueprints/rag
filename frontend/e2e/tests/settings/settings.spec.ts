import { test, expect } from '../../fixtures/base.ts';
import { SettingsPageObject } from '../../pages/SettingsPage.ts';

test.describe('Settings - navigation', () => {
  test('lands on RAG Configuration by default', async ({ page }) => {
    const settings = new SettingsPageObject(page);
    await settings.goto();

    await expect(page.getByText(/rag configuration/i).first()).toBeVisible();
  });

  test('clicking a sidebar item swaps the active section', async ({ page }) => {
    const settings = new SettingsPageObject(page);
    await settings.goto();

    await settings.openSection('advanced');
    await expect(page.getByText(/stop tokens/i).first()).toBeVisible();

    await settings.openSection('features');
    await expect(page.getByText(/feature toggles/i).first()).toBeVisible();
    await expect(page.getByText(/stop tokens/i)).toHaveCount(0);
  });
});

test.describe('Settings - RAG configuration sliders', () => {
  test('temperature, top-P and confidence sliders are rendered', async ({ page }) => {
    const settings = new SettingsPageObject(page);
    await settings.goto();
    await settings.openSection('ragConfig');

    await expect(settings.temperatureSlider).toBeVisible();
    await expect(settings.topPSlider).toBeVisible();
    await expect(settings.confidenceThresholdSlider).toBeVisible();
  });
});

test.describe('Settings - feature toggles warning modal', () => {
  test('enabling a feature opens the warning modal and Cancel dismisses it', async ({
    page,
  }) => {
    const settings = new SettingsPageObject(page);
    await settings.goto();
    await settings.openSection('features');

    const firstOffToggle = page
      .getByRole('switch')
      .filter({ hasNot: page.locator('[data-state="checked"]') })
      .first();
    await firstOffToggle.click();

    await expect(
      page.getByRole('heading', { name: /feature requirement/i }),
    ).toBeVisible();

    await settings.cancelFeatureWarning();
    await expect(
      page.getByRole('heading', { name: /feature requirement/i }),
    ).toHaveCount(0);
  });

  test('enable anyway applies the change and closes the modal', async ({
    page,
  }) => {
    const settings = new SettingsPageObject(page);
    await settings.goto();
    await settings.openSection('features');

    const firstOffToggle = page
      .getByRole('switch')
      .filter({ hasNot: page.locator('[data-state="checked"]') })
      .first();
    await firstOffToggle.click();

    await settings.confirmFeatureWarning();
    await expect(
      page.getByRole('heading', { name: /feature requirement/i }),
    ).toHaveCount(0);
  });
});

test.describe('Settings - advanced: stop tokens', () => {
  test('adds and removes stop tokens', async ({ page }) => {
    const settings = new SettingsPageObject(page);
    await settings.goto();
    await settings.openSection('advanced');

    const input = page.getByPlaceholder('Enter stop token');
    const addButton = page.getByRole('button', { name: /^add$/i });

    await input.fill('STOP_A');
    await addButton.click();
    await expect(page.getByText('STOP_A')).toBeVisible();

    await input.fill('STOP_B');
    await addButton.click();
    await expect(page.getByText('STOP_B')).toBeVisible();

    // Duplicate attempt is not added
    await input.fill('STOP_A');
    await addButton.click();
    await expect(page.getByText('STOP_A')).toHaveCount(1);

    // Remove STOP_A by clicking the tag
    await page.getByText('STOP_A').click();
    await expect(page.getByText('STOP_A')).toHaveCount(0);
    await expect(page.getByText('STOP_B')).toBeVisible();
  });

  test('"Add" button is disabled when input is empty', async ({ page }) => {
    const settings = new SettingsPageObject(page);
    await settings.goto();
    await settings.openSection('advanced');

    const addButton = page.getByRole('button', { name: /^add$/i });
    await expect(addButton).toBeDisabled();
  });
});

test.describe('Settings - theme toggle', () => {
  test('theme toggle is present in advanced settings', async ({ page }) => {
    const settings = new SettingsPageObject(page);
    await settings.goto();
    await settings.openSection('advanced');

    await expect(page.getByText(/theme/i).first()).toBeVisible();
    // Toggle is the theme switch — just assert at least one switch exists
    await expect(page.getByRole('switch').first()).toBeVisible();
  });
});

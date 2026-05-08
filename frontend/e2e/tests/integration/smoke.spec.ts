import { test, expect } from '../../fixtures/base.ts';
import { ChatPage } from '../../pages/ChatPage.ts';
import { SettingsPageObject } from '../../pages/SettingsPage.ts';

/**
 * Integration smoke tests. These run ONLY in the `integration` project,
 * which expects the backend stack (RAG orchestrator on :8081, Ingestor on :8082,
 * frontend dev server on :3000) to be running with real services.
 *
 * Run with:
 *   E2E_MODE=integration pnpm exec playwright test --project=integration
 */

test.describe('Integration - smoke', () => {
  test('app boots and renders core shell', async ({ page }) => {
    const chat = new ChatPage(page);
    await chat.goto();

    await expect(chat.pageRoot).toBeVisible();
    await expect(page.getByTestId('app-header')).toBeVisible();
    await expect(page.getByTestId('notification-bell')).toBeVisible();
  });

  test('GET /api/health responds successfully', async ({ page, request }) => {
    const response = await request.get('/api/health');
    expect(response.ok(), `unexpected /api/health status: ${response.status()}`).toBeTruthy();

    const body = await response.json();
    expect(body).toBeTruthy();
    // Just verify the response is a valid object.
    expect(typeof body).toBe('object');

    // Then sanity check that the UI boots with the same backend.
    await page.goto('/');
    await expect(page.getByTestId('chat-page')).toBeVisible();
  });

  test('settings page loads and responds to navigation', async ({ page }) => {
    const settings = new SettingsPageObject(page);
    await settings.goto();
    await settings.openSection('features');
    await expect(page.getByText(/feature toggles/i).first()).toBeVisible();
  });

  test('collections endpoint returns a valid list shape', async ({ request }) => {
    const response = await request.get('/api/collections');
    expect(response.ok()).toBeTruthy();
    const body = await response.json();
    expect(body).toBeTruthy();
    expect(Array.isArray(body.collections)).toBe(true);
  });
});

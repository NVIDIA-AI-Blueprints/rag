/**
 * Base test fixture. Extends @playwright/test with:
 *  - mockApi: auto-installed before any nav; per-test overrides via methods
 *  - seedStorage: deterministic localStorage seeder
 *  - axe: accessibility helper
 *  - coverage: V8 coverage start/stop (when E2E_COVERAGE=1)
 *
 * Import `test` and `expect` from this module in every spec instead of
 * `@playwright/test` to get the fixtures.
 */
// Playwright fixtures take a `use` callback that ESLint's react-hooks plugin
// mistakes for a React hook call. These are Playwright fixtures, not hooks —
// the plugin's rules don't apply.
/* eslint-disable react-hooks/rules-of-hooks */
import { test as base, expect } from '@playwright/test';
import { installMockApi, type MockApi } from './mock-api.ts';
import { seedStorage, type StorageSeed } from './storage.ts';
import { createAxeHelper, type AxeHelper } from './a11y.ts';
import { startCoverage, stopCoverage, coverageEnabled } from './coverage.ts';

export interface E2eFixtures {
  mockApi: MockApi;
  seedStorage: (seed?: StorageSeed) => Promise<void>;
  axe: AxeHelper;
  /**
   * Marker fixture indicating this test is running against the real backend
   * instead of mocks. Tests can branch on this if needed.
   */
  integrationMode: boolean;
}

export const test = base.extend<E2eFixtures>({
  integrationMode: [
    // eslint-disable-next-line no-empty-pattern
    async ({}, use) => {
      await use(process.env.E2E_MODE === 'integration');
    },
    { scope: 'test' },
  ],

  mockApi: [
    async ({ page, integrationMode }, use) => {
      if (integrationMode) {
        // No mocks in integration mode; return a no-op stub.
        const noop: MockApi = {
          setCollections() {},
          setCollectionDocuments() {},
          setHealth() {},
          setConfiguration() {},
          streamChat() {},
          releaseChat() {},
          failChat() {},
          setTaskProgression() {},
          setSummary() {},
          async route() {},
          requests: () => [],
          lastGenerateRequest: () => undefined,
        };
        await use(noop);
        return;
      }
      const api = await installMockApi(page);
      await use(api);
    },
    { auto: true },
  ],

  seedStorage: async ({ page }, use) => {
    await use((seed) => seedStorage(page, seed));
  },

  axe: async ({ page }, use, testInfo) => {
    await use(createAxeHelper(page, testInfo));
  },

  page: async ({ page }, use, testInfo) => {
    if (coverageEnabled) {
      await startCoverage(page);
    }
    await use(page);
    if (coverageEnabled) {
      await stopCoverage(page, testInfo);
    }
  },
});

export { expect };

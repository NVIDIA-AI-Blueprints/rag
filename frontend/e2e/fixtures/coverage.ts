/**
 * V8 coverage collector. Enabled only when E2E_COVERAGE=1.
 *
 * Coverage is forwarded to monocart-reporter via `addCoverageReport`, which
 * merges per-test V8 entries into the global coverage report configured in
 * playwright.config.ts.
 */
import type { Page, TestInfo } from '@playwright/test';
// monocart-reporter ships its own type declarations for this helper.
import { addCoverageReport } from 'monocart-reporter';

export const coverageEnabled = process.env.E2E_COVERAGE === '1';

function isChromium(page: Page): boolean {
  return page.context().browser()?.browserType().name() === 'chromium';
}

export async function startCoverage(page: Page): Promise<void> {
  if (!coverageEnabled) return;
  if (!isChromium(page)) return;
  await Promise.all([
    page.coverage.startJSCoverage({ resetOnNavigation: false }),
    page.coverage.startCSSCoverage({ resetOnNavigation: false }),
  ]);
}

export async function stopCoverage(page: Page, testInfo: TestInfo): Promise<void> {
  if (!coverageEnabled) return;
  if (!isChromium(page)) return;
  try {
    const [jsCoverage, cssCoverage] = await Promise.all([
      page.coverage.stopJSCoverage(),
      page.coverage.stopCSSCoverage(),
    ]);
    const merged = [...jsCoverage, ...cssCoverage];
    await addCoverageReport(merged, testInfo);
  } catch {
    // Coverage stop can fail if the page navigates mid-test; ignore.
  }
}


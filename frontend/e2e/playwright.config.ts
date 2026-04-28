import { defineConfig, devices } from '@playwright/test';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const isCI = !!process.env.CI;
const mode = process.env.E2E_MODE ?? 'mocked';
const baseURL = process.env.E2E_BASE_URL ?? 'http://localhost:3000';

const reporter: import('@playwright/test').ReporterDescription[] = [
  ['list'],
  ['html', { open: 'never', outputFolder: 'playwright-report' }],
];

if (isCI) {
  reporter.push(['blob']);
  reporter.push(['github']);
}

if (process.env.E2E_COVERAGE === '1') {
  reporter.push([
    'monocart-reporter',
    {
      name: 'RAG Blueprint E2E Coverage',
      outputFile: 'coverage/index.html',
      coverage: {
        entryFilter: (entry: { url: string }) => {
          if (
            !entry.url.startsWith('http://localhost:3000') &&
            !entry.url.startsWith('http://127.0.0.1:3000')
          ) {
            return false;
          }
          if (entry.url.includes('/node_modules/')) return false;
          if (entry.url.includes('/@vite/')) return false;
          if (entry.url.includes('/@react-refresh')) return false;
          if (entry.url.includes('/@id/')) return false;
          if (entry.url.includes('/@fs/')) return false;
          if (entry.url.endsWith('.css')) return false;
          return true;
        },
        sourceFilter: (sourcePath: string) => {
          if (sourcePath.includes('node_modules')) return false;
          if (sourcePath.includes('__tests__')) return false;
          if (sourcePath.endsWith('.test.ts') || sourcePath.endsWith('.test.tsx')) return false;
          return true;
        },
        // E2E-tier thresholds (deliberately lower than unit-test gates).
        // Industry guidance for E2E suites in 2026: lines/statements 50–65%,
        // functions 40–55%, branches 20–35% — branches are noisy in E2E since
        // they exercise mostly happy paths. These floors sit just below the
        // current measured coverage so they prevent regression today and can
        // be ratcheted upward as suites grow. Flip with COVERAGE_ENFORCE=1.
        thresholds: {
          lines: 50,
          statements: 50,
          functions: 40,
          branches: 25,
        },
        onEnd: (
          coverageResults: {
            summary: {
              lines: { pct: number };
              statements: { pct: number };
              functions: { pct: number };
              branches: { pct: number };
            };
          },
        ) => {
          const enforce = process.env.COVERAGE_ENFORCE === '1';
          if (!enforce) return;
          const { summary } = coverageResults;
          const t = { lines: 50, statements: 50, functions: 40, branches: 25 };
          const failed: string[] = [];
          if (summary.lines.pct < t.lines)
            failed.push(`lines ${summary.lines.pct}% < ${t.lines}%`);
          if (summary.statements.pct < t.statements)
            failed.push(`statements ${summary.statements.pct}% < ${t.statements}%`);
          if (summary.functions.pct < t.functions)
            failed.push(`functions ${summary.functions.pct}% < ${t.functions}%`);
          if (summary.branches.pct < t.branches)
            failed.push(`branches ${summary.branches.pct}% < ${t.branches}%`);
          if (failed.length) {
            throw new Error(`E2E coverage thresholds failed:\n  - ${failed.join('\n  - ')}`);
          }
        },
        reports: [
          ['v8'],
          ['console-summary'],
          ['lcovonly', { file: 'lcov.info' }],
          ['html-spa', { subdir: 'istanbul' }],
        ],
      },
    },
  ]);
}

const runsAgainstRealBackend = mode === 'integration';

export default defineConfig({
  testDir: './tests',
  fullyParallel: true,
  forbidOnly: isCI,
  // Retries are now opt-in per-test via the `@flaky` tag (see project filters
  // below). The 2026 consensus is that a global retry budget masks real
  // regressions: a genuine bug that happens to retry-pass once gets shipped.
  // Stable tests fail fast on first regression; tagged-flaky tests get a
  // bounded retry budget while their underlying flake is fixed (or quarantined).
  retries: 0,
  workers: isCI ? 4 : undefined,
  timeout: 30_000,
  expect: { timeout: 10_000 },
  reporter,
  outputDir: path.resolve(__dirname, 'test-results'),
  snapshotPathTemplate: '{testDir}/__snapshots__/{testFilePath}/{arg}{ext}',
  globalSetup: path.resolve(__dirname, 'global-setup.ts'),
  globalTeardown: path.resolve(__dirname, 'global-teardown.ts'),

  use: {
    baseURL,
    trace: 'on-first-retry',
    video: 'retain-on-failure',
    screenshot: 'only-on-failure',
    actionTimeout: 10_000,
    navigationTimeout: 20_000,
    testIdAttribute: 'data-testid',
  },

  projects: [
    // Stable tests: zero retries — first failure fails the build.
    {
      name: 'chromium',
      testIgnore: ['**/integration/**', '**/visual/**'],
      grepInvert: /@flaky/,
      use: { ...devices['Desktop Chrome'] },
    },
    {
      name: 'webkit',
      testIgnore: ['**/integration/**', '**/visual/**'],
      grepInvert: /@flaky/,
      use: { ...devices['Desktop Safari'] },
    },
    // Quarantined / known-flaky tests: bounded retry budget while the
    // underlying flake is being fixed. Tag a test as `@flaky` to opt it in:
    //
    //   test('something noisy @flaky', async ({ page }) => { ... });
    //
    // Quarantined tests should always have an owner + ticket + deadline
    // tracked outside the test (see e2e/README.md → "Quarantine workflow").
    {
      name: 'flaky-chromium',
      testIgnore: ['**/integration/**', '**/visual/**'],
      grep: /@flaky/,
      retries: isCI ? 2 : 1,
      use: { ...devices['Desktop Chrome'] },
    },
    {
      name: 'firefox',
      testIgnore: ['**/integration/**', '**/visual/**'],
      grep: /@cross-browser/,
      use: { ...devices['Desktop Firefox'] },
    },
    {
      name: 'visual',
      testMatch: '**/visual/**/*.spec.ts',
      use: { ...devices['Desktop Chrome'] },
    },
    {
      name: 'integration',
      testMatch: '**/integration/**/*.spec.ts',
      // Real backend can be flaky — give integration smoke a small retry budget.
      retries: isCI ? 2 : 0,
      use: { ...devices['Desktop Chrome'] },
    },
  ],

  webServer: runsAgainstRealBackend
    ? undefined
    : {
        command: 'pnpm --dir .. dev',
        url: baseURL,
        reuseExistingServer: !isCI,
        timeout: 120_000,
        stdout: 'ignore',
        stderr: 'pipe',
        env: {
          VITE_API_CHAT_URL: 'http://127.0.0.1:9/v1',
          VITE_API_VDB_URL: 'http://127.0.0.1:9/v1',
        },
      },
});

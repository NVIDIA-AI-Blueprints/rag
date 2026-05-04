/**
 * Accessibility helper built on @axe-core/playwright.
 *
 * Use `axe.scan()` for a full-page scan. Chain `.include(...)` / `.exclude(...)`
 * via `axe.builder()` when scoping is required.
 */
import type { Page, TestInfo } from '@playwright/test';
import { AxeBuilder } from '@axe-core/playwright';

export interface AxeHelper {
  scan(options?: { include?: string; exclude?: string[] }): Promise<void>;
  builder(): AxeBuilder;
}

// Rules we intentionally skip — KUI injects some internal wrappers where
// these are best reviewed manually rather than blocking CI.
const DEFAULT_DISABLED_RULES = [
  // KUI design tokens already audited; disable only if false positives arise.
];

// Known benign third-party selectors (extend as needed).
const DEFAULT_EXCLUDE: string[] = [];

export function createAxeHelper(page: Page, testInfo: TestInfo): AxeHelper {
  const buildBase = () => {
    let builder = new AxeBuilder({ page }).withTags([
      'wcag2a',
      'wcag2aa',
      'wcag21a',
      'wcag21aa',
      'best-practice',
    ]);
    for (const rule of DEFAULT_DISABLED_RULES) {
      builder = builder.disableRules(rule);
    }
    for (const selector of DEFAULT_EXCLUDE) {
      builder = builder.exclude(selector);
    }
    return builder;
  };

  return {
    builder: buildBase,
    async scan({ include, exclude = [] } = {}) {
      let builder = buildBase();
      if (include) builder = builder.include(include);
      for (const sel of exclude) builder = builder.exclude(sel);
      const results = await builder.analyze();
      await testInfo.attach('axe-results', {
        body: JSON.stringify(results, null, 2),
        contentType: 'application/json',
      });
      const serious = results.violations.filter(
        (v) => v.impact === 'critical' || v.impact === 'serious',
      );
      if (serious.length > 0) {
        const summary = serious
          .map((v) => `- [${v.impact}] ${v.id}: ${v.help} (${v.nodes.length} nodes)`)
          .join('\n');
        throw new Error(
          `Accessibility violations found (critical/serious only):\n${summary}`,
        );
      }
    },
  };
}

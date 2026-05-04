# End-to-End Tests

Playwright end-to-end test suite for the RAG Blueprint frontend. Covers the
chat, collections, settings, navigation, notifications, accessibility, and
visual regression flows, with an opt-in smoke suite for the real backend.

## Quick start

```bash
# From the frontend/ directory
pnpm install
pnpm e2e:install                           # downloads chromium/webkit/firefox + system deps
pnpm e2e                                   # runs chromium + webkit (mocked)
```

The Playwright config automatically starts the Vite dev server on
`http://localhost:3000` and points the app's chat/VDB endpoints to an
unreachable address (`127.0.0.1:9`) so any unmocked network call fails loudly.

## Run modes

| Mode        | What it does                                                             |
|-------------|--------------------------------------------------------------------------|
| Mocked      | Default. All `/api/*` requests are intercepted by `fixtures/mock-api.ts`. |
| Integration | `E2E_MODE=integration`. Hits the real RAG orchestrator + Ingestor.       |

Mocked tests are fast (~15 s on chromium, ~30 s with webkit), deterministic,
and run in CI on every PR. Integration tests are opt-in — run them manually
before a release or when backend API contracts change. The integration project
expects the backend already to be reachable at `E2E_BASE_URL` (the webServer
hook is disabled in integration mode).

## Useful scripts

All scripts live in `frontend/package.json` and can be invoked from the
`frontend/` directory:

```bash
pnpm e2e                       # chromium + webkit (default mocked suite)
pnpm e2e:ui                    # Playwright UI runner
pnpm e2e:headed                # headed browser for debugging
pnpm e2e:debug                 # debug mode with inspector
pnpm e2e:chromium              # chromium only
pnpm e2e:webkit                # webkit only
pnpm e2e:firefox               # firefox project — runs only specs tagged @cross-browser
pnpm e2e:a11y                  # axe-core accessibility scans (chromium)
pnpm e2e:visual                # visual regression (chromium, requires baselines)
pnpm e2e:visual:update         # regenerate visual baselines
pnpm e2e:integration           # run against the real backend (requires it running)
pnpm e2e:coverage              # collect V8 + CSS coverage via monocart-reporter
pnpm e2e:report                # open the last HTML report
pnpm e2e:install               # install Playwright browsers + system deps
```

> **Firefox note**: the `firefox` project filters on `@cross-browser` tags
> (see `playwright.config.ts`). No specs currently carry that tag, so
> `pnpm e2e:firefox` is effectively a no-op until you tag tests as
> `test('does X @cross-browser', ...)` to opt them into Firefox runs.

Pass extra Playwright flags directly to the script:

```bash
pnpm e2e --grep "streaming" --project=chromium
```

## Project structure

```
e2e/
├── fixtures/          # typed test fixtures (mockApi, seededStorage, axe, coverage)
│   ├── base.ts
│   ├── mock-api.ts
│   ├── storage.ts
│   ├── a11y.ts
│   └── coverage.ts
├── pages/             # Page Object Models (user-intent APIs over semantic locators)
│   ├── BasePage.ts
│   ├── ChatPage.ts
│   ├── CollectionsPanel.ts
│   ├── NewCollectionPage.ts
│   ├── SettingsPage.ts
│   ├── NotificationPanel.ts
│   └── CitationsDrawer.ts
├── tests/             # specs grouped by feature area
│   ├── chat/
│   ├── collections/
│   ├── settings/
│   ├── navigation/
│   ├── notifications/
│   ├── a11y/
│   ├── visual/
│   └── integration/
├── utils/
│   ├── sse.ts         # helpers for mocking server-sent events
│   ├── fixtures-data.ts
│   ├── paths.ts
│   └── files/         # tiny sample upload fixtures (.png / .txt / .pdf)
├── global-setup.ts    # placeholder hook, wired in playwright.config.ts
├── global-teardown.ts # placeholder hook, wired in playwright.config.ts
├── playwright.config.ts
├── tsconfig.json
└── README.md
```

## Writing tests

1. **Always import from `fixtures/base.ts`** to get `mockApi`, `seedStorage`,
   `axe`, and the (optional) coverage fixture:

   ```ts
   import { test, expect } from '../../fixtures/base.ts';
   ```

2. **Prefer Page Object Models** over raw selectors. POMs encode user intent
   (e.g. `chat.askQuestion("…")`) and are resilient to DOM changes.

3. **Use `data-testid` first, semantic role second.** The config sets
   `testIdAttribute: 'data-testid'`, so `page.getByTestId(...)` resolves
   `data-testid="..."`.

4. **Per-test mock overrides** live on the `mockApi` fixture:

   ```ts
   mockApi.setCollections(['docs']);
   mockApi.streamChat({ text: 'Hello', chunks: 6 });
   mockApi.failChat(500, '{"detail":"boom"}');
   mockApi.setHealth('error');
   ```

5. **Seed `localStorage`** before navigation via `seedStorage`. This runs
   inside `page.addInitScript` so the app sees it on first paint.

## Coverage

Coverage is opt-in — set `E2E_COVERAGE=1` (or run `pnpm e2e:coverage`) and
Playwright will collect V8 + CSS coverage and forward it to
`monocart-reporter` via `addCoverageReport`. Output goes to
`frontend/coverage/` (HTML, lcov, raw v8).

### Thresholds

E2E coverage is intentionally measured at a **lower bar** than unit-test
coverage: E2E exercises user journeys, not every code branch. Industry
guidance (test-pyramid, 2026 surveys) puts typical E2E floors at:

| Metric     | Unit-test default | E2E default (this repo) |
|------------|------------------:|------------------------:|
| Lines      | 80 %              | **50 %**                |
| Statements | 80 %              | **50 %**                |
| Functions  | 75 %              | **40 %**                |
| Branches   | 70 % (often 10–50 %) | **25 %**             |

Current measured E2E coverage on `chromium` (59 mocked specs):

| Metric     | Coverage |
|------------|---------:|
| Lines      | 55 %     |
| Statements | 58 %     |
| Functions  | 47 %     |
| Branches   | 29 %     |

The thresholds sit just below current numbers so they prevent regression
today and can be ratcheted upward as the suite grows. They are enforced
only when `COVERAGE_ENFORCE=1` is set (e.g. in CI):

```bash
E2E_COVERAGE=1 COVERAGE_ENFORCE=1 pnpm e2e:chromium
```

Treat the unit-test suite (`pnpm test:coverage` via Vitest) as the
source-of-truth for high coverage gates. The E2E numbers should be
read as a **journey-coverage** signal: what proportion of the rendered
React app is exercised by realistic user flows.

## CI

See `.github/workflows/e2e.yml` for the sharded CI workflow:

- **`e2e-mocked`**: runs on every PR touching `frontend/**`. Parallelized
  across 4 shards; merges blob reports into a single HTML report artifact.
- **`e2e-a11y-visual`**: runs axe scans (blocking) and visual regression
  (non-blocking) on Chromium only.
- **`e2e-integration`**: opt-in via `workflow_dispatch` with `mode=integration`.
  Requires backend services to be reachable at `E2E_BASE_URL`.

## Debugging failures

- **HTML report** (local): `pnpm e2e:report`
- **Trace viewer** (local): `pnpm exec playwright show-trace e2e/test-results/<test>/trace.zip`
- **CI**: download the `playwright-report-<shard>` or `test-results-<shard>`
  artifacts from the failed job, then run `pnpm exec playwright show-report ./playwright-report-<shard>`.

## Troubleshooting

- **"E2E mock missing for …"** — the catch-all 503 fixture fired because a
  route wasn't set up. Either set a default response in `mock-api.ts` or
  override it in the test with `page.route`.
- **Blank page on first load** — double-check any `page.route` overrides in
  specs; they must not match Vite's module import paths (e.g. `/src/api/…`).
  The built-in mocks use regex patterns anchored to `/api/…` to avoid this.
- **Flaky SSE assertions** — use `buildStreamBody` (`utils/sse.ts`) to generate
  well-formed SSE responses. Avoid sleeping in the mock; prefer chunked
  responses and Playwright's auto-waiting.
- **`aria-label` assertions fail** — accessibility attributes sometimes
  rehydrate after React Query settles. Assert after `waitForAppReady()` and
  use `toHaveAttribute('aria-label', /pattern/)` rather than snapshotting.

/**
 * Mock API fixture — intercepts every /api/* endpoint the frontend uses.
 *
 * Default behavior: healthy, empty-ish responses so that the app boots and
 * the send button is enabled. Per-test overrides replace individual handlers.
 *
 * Unmocked calls fall through to a 503 handler that fails the test loudly,
 * so forgetting a mock is always obvious.
 */
import type { Page, Route, Request } from '@playwright/test';
import {
  buildCollectionDocuments,
  buildConfiguration,
  buildHealthy,
  buildUnhealthy,
  buildTaskFinished,
  buildTaskPending,
  type MockCollection,
} from '../utils/fixtures-data.ts';
import {
  buildStreamBody,
  type BuildStreamOptions,
} from '../utils/sse.ts';

type HealthState = 'healthy' | 'unhealthy' | 'error';

interface StreamConfig extends BuildStreamOptions {
  /**
   * Optional HTTP status. Default 200. Set to a 4xx/5xx value to test error
   * paths. If 0, network aborts (simulating connection drop).
   */
  status?: number;
  /**
   * Wall-clock delay before the response is sent (ms). Default 0.
   *
   * Avoid this in new tests — wall-clock waits are flaky under load. Use
   * `holdChat()` + `releaseChat()` instead for deterministic control over
   * when the response completes.
   */
  delayMs?: number;
  /**
   * If true, the route handler awaits an internal Deferred until the test
   * calls `releaseChat()` (or aborts via `failChat`). Lets tests deterministically
   * assert on streaming state (e.g. stop button visible) without wall-clock waits.
   */
  hold?: boolean;
  /** Assertion hook invoked with the parsed request body. */
  onRequest?: (body: unknown, request: Request) => void;
}

type DocumentsStore = Map<
  string,
  Array<{ name: string; description?: string; tags?: string[] }>
>;

export interface MockApi {
  /** Set or clear the collections list. */
  setCollections(collections: string[] | MockCollection[]): void;
  /** Seed documents for a specific collection. */
  setCollectionDocuments(
    collection: string,
    documents: Array<{ name: string; description?: string; tags?: string[] }>,
  ): void;
  /** Override /api/health state. */
  setHealth(state: HealthState): void;
  /** Override /api/configuration response. */
  setConfiguration(overrides: Partial<ReturnType<typeof buildConfiguration>>): void;
  /** Configure the next /api/generate stream. */
  streamChat(config?: StreamConfig): void;
  /**
   * Release any in-flight `streamChat({ hold: true })` request, allowing the
   * mocked SSE response to complete. Safe to call when nothing is held.
   */
  releaseChat(): void;
  /** Return a queued chat response that returns 500. */
  failChat(status?: number, body?: string): void;
  /**
   * Configure /api/status to return a PENDING task N times then FINISHED.
   * Default: immediately FINISHED.
   */
  setTaskProgression(taskId: string, pendingCalls?: number): void;
  /**
   * Customize document summary endpoint.
   */
  setSummary(summaryText: string, state?: 'PENDING' | 'COMPLETED' | 'FAILED'): void;
  /**
   * Register a custom handler for any /api/* route pattern. Runs before defaults.
   */
  route(
    urlPattern: string | RegExp,
    handler: (route: Route, request: Request) => Promise<void> | void,
  ): Promise<void>;
  /** Snapshot of all requests observed so far. */
  requests(): Request[];
  /** Last recorded generate request body. */
  lastGenerateRequest(): unknown | undefined;
}

export async function installMockApi(page: Page): Promise<MockApi> {
  let collections: MockCollection[] = [];
  const documentsStore: DocumentsStore = new Map();
  let health: HealthState = 'healthy';
  let configuration = buildConfiguration();
  let streamConfig: StreamConfig = { text: 'Mocked response.' };
  let chatFailure: { status: number; body: string } | null = null;
  const taskProgressions = new Map<string, { remaining: number }>();
  let summary: { text: string; state: 'PENDING' | 'COMPLETED' | 'FAILED' } = {
    text: 'Mocked document summary.',
    state: 'COMPLETED',
  };

  const observed: Request[] = [];
  let lastGenerateBody: unknown;

  // Deferred used to deterministically hold an in-flight /api/generate response
  // open until the test calls `releaseChat()`. Replaces wall-clock `delayMs`
  // for stop-button / streaming-state assertions.
  let holdRelease: (() => void) | null = null;
  const newHoldGate = (): Promise<void> =>
    new Promise<void>((resolve) => {
      holdRelease = resolve;
    });

  // Disable CSS transitions/animations globally so that KUI Popover / SidePanel
  // / Dropdown enter-leave animations don't race assertions. This is the 2026
  // recommended workaround for the lack of a global Playwright "no animations"
  // flag (animations: 'disabled' only applies per-screenshot / per-action).
  await page.addInitScript(() => {
    const css = `*,*::before,*::after{animation-duration:0s !important;animation-delay:0s !important;transition-duration:0s !important;transition-delay:0s !important;scroll-behavior:auto !important}`;
    const apply = () => {
      if (document.head && !document.getElementById('e2e-disable-animations')) {
        const style = document.createElement('style');
        style.id = 'e2e-disable-animations';
        style.textContent = css;
        document.head.appendChild(style);
      }
    };
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', apply, { once: true });
    } else {
      apply();
    }
  });

  const recordAndContinue = async (
    route: Route,
    handler: (route: Route, request: Request) => Promise<void> | void,
  ) => {
    const request = route.request();
    observed.push(request);
    await handler(route, request);
  };

  // We match on real API paths only. Using a regex anchored to "/api/" at the
  // URL *path* boundary avoids hijacking Vite dev-server module imports like
  // "/src/api/useHealthApi.ts" which would otherwise also match `**/api/**`.
  //
  // Pattern: scheme://host[:port]/api/... with no "/src/" before it.
  const apiPath = (suffix: string) => new RegExp(`^https?://[^/]+/api/${suffix}`);

  // NOTE: Playwright matches routes LIFO (last-registered wins). We register
  // the catch-all *first* so that the specific handlers below override it.
  //
  // Fallback for any unmocked /api/* call — fails loudly so missing mocks
  // surface as obvious test failures.
  await page.route(apiPath('.*'), async (route) => {
    observed.push(route.request());
    await route.fulfill({
      status: 503,
      contentType: 'application/json',
      body: JSON.stringify({
        detail: `E2E mock missing for ${route.request().method()} ${route.request().url()}`,
      }),
    });
  });

  // /api/collections — GET lists, DELETE removes
  await page.route(apiPath('collections(\\?|$)'), async (route) => {
    await recordAndContinue(route, async (r, req) => {
      if (req.method() === 'GET') {
        await r.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ collections }),
        });
        return;
      }
      if (req.method() === 'DELETE') {
        try {
          const names = JSON.parse(req.postData() ?? '[]') as string[];
          collections = collections.filter(
            (c) => !names.includes(c.collection_name),
          );
        } catch {
          // ignore malformed body
        }
        await r.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ message: 'deleted' }),
        });
        return;
      }
      await r.fallback();
    });
  });

  // /api/collection (create)
  await page.route(apiPath('collection(\\?|$)'), async (route) => {
    await recordAndContinue(route, async (r, req) => {
      if (req.method() !== 'POST') {
        await r.fallback();
        return;
      }
      try {
        const payload = JSON.parse(req.postData() ?? '{}') as { collection_name?: string };
        if (payload.collection_name) {
          if (collections.some((c) => c.collection_name === payload.collection_name)) {
            await r.fulfill({
              status: 409,
              contentType: 'application/json',
              body: JSON.stringify({ detail: 'Collection already exists' }),
            });
            return;
          }
          collections.push({
            collection_name: payload.collection_name,
            num_entities: 0,
            metadata_schema: [],
          });
        }
      } catch {
        // ignore
      }
      await r.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ message: 'created' }),
      });
    });
  });

  // /api/documents — GET lists, POST uploads (returns task_id), DELETE removes
  await page.route(apiPath('documents'), async (route) => {
    await recordAndContinue(route, async (r, req) => {
      const url = new URL(req.url());
      const collection = url.searchParams.get('collection_name') ?? '';
      if (req.method() === 'GET') {
        const docs = documentsStore.get(collection) ?? [];
        await r.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(buildCollectionDocuments(docs)),
        });
        return;
      }
      if (req.method() === 'POST') {
        const taskId = `task-${Date.now()}`;
        taskProgressions.set(taskId, { remaining: 0 });
        await r.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ task_id: taskId, message: 'accepted' }),
        });
        return;
      }
      if (req.method() === 'DELETE') {
        await r.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ message: 'deleted' }),
        });
        return;
      }
      await r.fallback();
    });
  });

  // /api/status?task_id=
  await page.route(apiPath('status'), async (route) => {
    await recordAndContinue(route, async (r, req) => {
      const url = new URL(req.url());
      const taskId = url.searchParams.get('task_id') ?? 'task-1';
      const progression = taskProgressions.get(taskId);
      let task;
      if (progression && progression.remaining > 0) {
        progression.remaining -= 1;
        task = buildTaskPending(taskId);
      } else {
        task = buildTaskFinished(taskId);
      }
      await r.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(task),
      });
    });
  });

  // /api/health
  await page.route(apiPath('health'), async (route) => {
    await recordAndContinue(route, async (r) => {
      if (health === 'error') {
        await r.fulfill({ status: 500, body: 'internal error' });
        return;
      }
      const body = health === 'healthy' ? buildHealthy() : buildUnhealthy();
      await r.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(body),
      });
    });
  });

  // /api/configuration
  await page.route(apiPath('configuration'), async (route) => {
    await recordAndContinue(route, async (r) => {
      await r.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(configuration),
      });
    });
  });

  // /api/summary
  await page.route(apiPath('summary'), async (route) => {
    await recordAndContinue(route, async (r) => {
      await r.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          summary: summary.text,
          state: summary.state,
        }),
      });
    });
  });

  // /api/generate — SSE stream
  await page.route(apiPath('generate'), async (route) => {
    await recordAndContinue(route, async (r, req) => {
      try {
        lastGenerateBody = JSON.parse(req.postData() ?? '{}');
      } catch {
        lastGenerateBody = req.postData();
      }
      if (streamConfig.onRequest) {
        streamConfig.onRequest(lastGenerateBody, req);
      }

      if (chatFailure) {
        await r.fulfill({
          status: chatFailure.status,
          contentType: 'application/json',
          body: chatFailure.body,
        });
        return;
      }

      if (streamConfig.status === 0) {
        await r.abort('failed');
        return;
      }

      const body = buildStreamBody(streamConfig);

      // Deterministic hold: keep the response pending until releaseChat() is
      // called. Tests asserting on streaming state (stop button visible, etc.)
      // should use this instead of wall-clock delays.
      if (streamConfig.hold) {
        await newHoldGate();
      } else if (streamConfig.delayMs && streamConfig.delayMs > 0) {
        // Legacy wall-clock delay — kept for back-compat but discouraged.
        await new Promise((resolve) =>
          setTimeout(resolve, streamConfig.delayMs),
        );
      }
      await r.fulfill({
        status: streamConfig.status ?? 200,
        headers: {
          'content-type': 'text/event-stream',
          'cache-control': 'no-cache',
        },
        body,
      });
    });
  });

  // /api/collections/{name}/documents/{doc}/metadata PATCH
  await page.route(/^https?:\/\/[^/]+\/api\/collections\/[^/]+\/documents\/[^/]+\/metadata/, async (route) => {
    await recordAndContinue(route, async (r) => {
      await r.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ message: 'updated' }),
      });
    });
  });

  const api: MockApi = {
    setCollections(next) {
      if (next.length === 0) {
        collections = [];
        return;
      }
      if (typeof next[0] === 'string') {
        collections = (next as string[]).map((name) => ({
          collection_name: name,
          num_entities: 0,
          metadata_schema: [],
        }));
      } else {
        collections = [...(next as MockCollection[])];
      }
    },
    setCollectionDocuments(collection, docs) {
      documentsStore.set(collection, docs);
    },
    setHealth(state) {
      health = state;
    },
    setConfiguration(overrides) {
      configuration = { ...configuration, ...overrides } as ReturnType<
        typeof buildConfiguration
      >;
    },
    streamChat(config = {}) {
      chatFailure = null;
      streamConfig = { text: 'Mocked response.', ...config };
    },
    releaseChat() {
      if (holdRelease) {
        const fn = holdRelease;
        holdRelease = null;
        fn();
      }
    },
    failChat(status = 500, body = '{"detail":"mocked failure"}') {
      chatFailure = { status, body };
    },
    setTaskProgression(taskId, pendingCalls = 0) {
      taskProgressions.set(taskId, { remaining: pendingCalls });
    },
    setSummary(text, state = 'COMPLETED') {
      summary = { text, state };
    },
    async route(urlPattern, handler) {
      await page.route(urlPattern, async (route, request) => {
        observed.push(request);
        await handler(route, request);
      });
    },
    requests() {
      return [...observed];
    },
    lastGenerateRequest() {
      return lastGenerateBody;
    },
  };

  return api;
}

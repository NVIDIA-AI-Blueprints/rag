/**
 * Seeds deterministic localStorage before any app JS runs.
 * Covers the keys the app reads:
 *  - `rag-settings` (zustand persist)
 *  - `ingestion-task-*` / `completedTask:*` (NotificationStore hydration)
 *  - `useCollectionConfigStore` persist
 */
import type { Page } from '@playwright/test';

export interface StorageSeed {
  settings?: {
    useLocalStorage?: boolean;
    theme?: 'light' | 'dark';
    [key: string]: unknown;
  };
  pendingTasks?: Array<{
    id: string;
    collection_name: string;
    state: 'PENDING' | 'FINISHED' | 'FAILED' | 'UNKNOWN';
  }>;
  completedTasks?: Array<{
    id: string;
    collection_name: string;
    state: 'FINISHED' | 'FAILED';
  }>;
}

const DEFAULT_SETTINGS = {
  useLocalStorage: false,
  theme: 'dark' as const,
};

export async function seedStorage(page: Page, seed: StorageSeed = {}): Promise<void> {
  const settings = { ...DEFAULT_SETTINGS, ...(seed.settings ?? {}) };
  const pending = seed.pendingTasks ?? [];
  const completed = seed.completedTasks ?? [];

  await page.addInitScript(
    ({ settings, pending, completed }) => {
      try {
        if (settings.useLocalStorage) {
          window.localStorage.setItem(
            'rag-settings',
            JSON.stringify({ state: settings, version: 0 }),
          );
        } else {
          window.localStorage.removeItem('rag-settings');
        }

        // Wipe any stale notification keys
        for (let i = window.localStorage.length - 1; i >= 0; i--) {
          const key = window.localStorage.key(i);
          if (key && (key.startsWith('ingestion-task-') || key.startsWith('completedTask:'))) {
            window.localStorage.removeItem(key);
          }
        }

        for (const task of pending) {
          window.localStorage.setItem(
            `ingestion-task-${task.id}`,
            JSON.stringify({
              id: task.id,
              collection_name: task.collection_name,
              state: task.state,
              created_at: new Date().toISOString(),
            }),
          );
        }
        for (const task of completed) {
          window.localStorage.setItem(
            `completedTask:${task.id}`,
            JSON.stringify({
              id: task.id,
              collection_name: task.collection_name,
              state: task.state,
              created_at: new Date().toISOString(),
            }),
          );
        }
      } catch {
        // SecurityError in some contexts; tests that need storage will fail visibly.
      }
    },
    { settings, pending, completed },
  );
}

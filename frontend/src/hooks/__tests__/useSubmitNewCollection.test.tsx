// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { ReactNode } from 'react';

const mockNavigate = vi.fn();
vi.mock('react-router-dom', () => ({
  useNavigate: () => mockNavigate,
}));

const mockShowToast = vi.fn();
vi.mock('../../store/useToastStore', () => ({
  useToastStore: () => ({ showToast: mockShowToast }),
}));

const mockAddTaskNotification = vi.fn();
vi.mock('../../store/useNotificationStore', () => ({
  useNotificationStore: () => ({ addTaskNotification: mockAddTaskNotification }),
}));

const mockSetIsLoading = vi.fn();
const mockSetUploadComplete = vi.fn();
const mockSetError = vi.fn();
const mockReset = vi.fn();

interface NewCollectionStoreState {
  collectionName: string;
  metadataSchema: unknown[];
  fileMetadata: Record<string, unknown>;
  selectedFiles: File[];
  catalogMetadata: { tags: string[]; description?: string; owner?: string; created_by?: string; business_domain?: string; status?: string };
  collectionConfig: { generateSummary: boolean };
  setIsLoading: typeof mockSetIsLoading;
  setUploadComplete: typeof mockSetUploadComplete;
  setError: typeof mockSetError;
  reset: typeof mockReset;
}

const newCollectionState: NewCollectionStoreState = {
  collectionName: 'test-collection',
  metadataSchema: [],
  fileMetadata: {},
  selectedFiles: [new File(['hello'], 'test.txt', { type: 'text/plain' })],
  catalogMetadata: { tags: [] },
  collectionConfig: { generateSummary: false },
  setIsLoading: mockSetIsLoading,
  setUploadComplete: mockSetUploadComplete,
  setError: mockSetError,
  reset: mockReset,
};

vi.mock('../../store/useNewCollectionStore', () => ({
  useNewCollectionStore: () => newCollectionState,
}));

vi.mock('../../store/useSettingsStore', () => ({
  useSettingsStore: () => ({ vdbEndpoint: '' }),
}));

vi.mock('../../components/notifications/NotificationBell', () => ({
  openNotificationPanel: vi.fn(),
}));

import { useSubmitNewCollection } from '../useSubmitNewCollection';

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
}

describe('useSubmitNewCollection — error notification surfacing', () => {
  let fetchSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    vi.clearAllMocks();
    fetchSpy = vi.spyOn(global, 'fetch');
    newCollectionState.selectedFiles = [new File(['hello'], 'test.txt', { type: 'text/plain' })];
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('shows a toast when collection creation returns HTTP 503 (Elasticsearch unavailable)', async () => {
    fetchSpy.mockResolvedValueOnce({
      ok: false,
      status: 503,
      json: async () => ({
        message:
          'Vector database (Elasticsearch) is unavailable at http://elasticsearch:9200. Please verify Elasticsearch is running and accessible.',
      }),
    } as Response);

    const { result } = renderHook(() => useSubmitNewCollection(), { wrapper: createWrapper() });

    await act(async () => {
      await result.current.submit();
    });

    expect(mockShowToast).toHaveBeenCalledWith(
      'Vector database (Elasticsearch) is unavailable at http://elasticsearch:9200. Please verify Elasticsearch is running and accessible.',
      'error',
    );
    expect(mockSetError).toHaveBeenCalledWith(
      'Vector database (Elasticsearch) is unavailable at http://elasticsearch:9200. Please verify Elasticsearch is running and accessible.',
    );
    expect(mockNavigate).not.toHaveBeenCalled();
  });

  it('shows a toast with the Pydantic detail message stripped of "Value error, " on HTTP 422', async () => {
    fetchSpy.mockResolvedValueOnce({
      ok: false,
      status: 422,
      json: async () => ({
        detail: [{ msg: 'Value error, Collection name must be lowercase', type: 'value_error', loc: ['body'] }],
      }),
    } as Response);

    const { result } = renderHook(() => useSubmitNewCollection(), { wrapper: createWrapper() });

    await act(async () => {
      await result.current.submit();
    });

    expect(mockShowToast).toHaveBeenCalledWith('Collection name must be lowercase', 'error');
  });

  it('shows a toast when document upload returns HTTP 500 after collection was created', async () => {
    fetchSpy.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ message: 'Collection test-collection created successfully.' }),
    } as Response);
    fetchSpy.mockResolvedValueOnce({
      ok: false,
      status: 500,
      json: async () => ({
        message:
          'Ingestion of files failed with error: Collection test-collection does not exist. Ensure a collection is created using POST /collection endpoint first.',
      }),
    } as Response);

    const { result } = renderHook(() => useSubmitNewCollection(), { wrapper: createWrapper() });

    await act(async () => {
      await result.current.submit();
    });

    expect(mockShowToast).toHaveBeenCalledWith(
      expect.stringContaining('Collection test-collection does not exist'),
      'error',
    );
    expect(mockNavigate).not.toHaveBeenCalled();
  });

  it('falls back to default error message when response body is not JSON', async () => {
    fetchSpy.mockResolvedValueOnce({
      ok: false,
      status: 502,
      json: async () => {
        throw new Error('Unexpected token < in JSON');
      },
    } as Response);

    const { result } = renderHook(() => useSubmitNewCollection(), { wrapper: createWrapper() });

    await act(async () => {
      await result.current.submit();
    });

    expect(mockShowToast).toHaveBeenCalledWith('Failed to create collection', 'error');
  });

  it('shows the upload "Failed to upload documents" fallback when upload response is non-JSON', async () => {
    fetchSpy.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ message: 'Collection test-collection created successfully.' }),
    } as Response);
    fetchSpy.mockResolvedValueOnce({
      ok: false,
      status: 502,
      json: async () => {
        throw new Error('Unexpected token < in JSON');
      },
    } as Response);

    const { result } = renderHook(() => useSubmitNewCollection(), { wrapper: createWrapper() });

    await act(async () => {
      await result.current.submit();
    });

    expect(mockShowToast).toHaveBeenCalledWith('Failed to upload documents', 'error');
  });
});

// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { ReactNode } from 'react';
import { useCreateCollection, useCollections, useDeleteCollection } from '../../api/useCollectionsApi';

// Create a wrapper with QueryClient for hooks
function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
      mutations: {
        retry: false,
      },
    },
  });

  const Wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={queryClient}>
      {children}
    </QueryClientProvider>
  );
  
  return Wrapper;
}

describe('useCollectionsApi', () => {
  let fetchSpy: any;

  beforeEach(() => {
    fetchSpy = vi.spyOn(global, 'fetch');
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('useCollections', () => {
    it('fetches collections successfully', async () => {
      const mockCollections = [
        { collection_name: 'collection1', embedding_dimension: 1536 },
        { collection_name: 'collection2', embedding_dimension: 768 },
      ];

      fetchSpy.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ collections: mockCollections }),
      } as Response);

      const { result } = renderHook(() => useCollections(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => expect(result.current.isSuccess).toBe(true));

      expect(result.current.data).toEqual(mockCollections);
      expect(fetchSpy).toHaveBeenCalledWith('/api/collections');
    });

    it('handles fetch error', async () => {
      fetchSpy.mockResolvedValueOnce({
        ok: false,
        status: 500,
      } as Response);

      const { result } = renderHook(() => useCollections(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => expect(result.current.isError).toBe(true));
      expect(result.current.error).toBeDefined();
    });
  });

  describe('useCreateCollection', () => {
    it('creates collection successfully', async () => {
      const mockResponse = { 
        collection_name: 'test-collection',
        status: 'created' 
      };

      fetchSpy.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      } as Response);

      const { result } = renderHook(() => useCreateCollection(), {
        wrapper: createWrapper(),
      });

      const payload = {
        collection_name: 'test-collection',
        embedding_dimension: 1536,
        metadata_schema: [],
        vdb_endpoint: 'http://localhost:8000',
      };

      result.current.mutate(payload);

      await waitFor(() => expect(result.current.isSuccess).toBe(true));

      expect(fetchSpy).toHaveBeenCalledWith('/api/collection', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      expect(result.current.data).toEqual(mockResponse);
    });

    // NEW TEST: Verify Pydantic validation error parsing (strips "Value error, " prefix)
    it('handles Pydantic validation error and strips "Value error," prefix', async () => {
      fetchSpy.mockResolvedValueOnce({
        ok: false,
        status: 422,
        json: async () => ({
          detail: [
            {
              msg: "Value error, Field name 'type' is reserved and cannot be used as a custom metadata field. Please choose a different field name.",
              type: 'value_error',
              loc: ['body', 'metadata_schema', 0, 'name'],
            },
          ],
        }),
      } as Response);

      const { result } = renderHook(() => useCreateCollection(), {
        wrapper: createWrapper(),
      });

      const payload = {
        collection_name: 'test-collection',
        embedding_dimension: 1536,
        metadata_schema: [{ name: 'type', type: 'string' as const }],
        vdb_endpoint: 'http://localhost:8000',
      };

      result.current.mutate(payload);

      await waitFor(() => expect(result.current.isError).toBe(true));

      // Should strip "Value error, " prefix
      expect(result.current.error?.message).toBe(
        "Field name 'type' is reserved and cannot be used as a custom metadata field. Please choose a different field name."
      );
    });

    // NEW TEST: Verify error message fallback when detail is missing
    it('falls back to default error message when detail parsing fails', async () => {
      fetchSpy.mockResolvedValueOnce({
        ok: false,
        status: 422,
        json: async () => ({
          detail: [
            {
              type: 'value_error',
              loc: ['body', 'metadata_schema', 0, 'name'],
              // msg field missing
            },
          ],
        }),
      } as Response);

      const { result } = renderHook(() => useCreateCollection(), {
        wrapper: createWrapper(),
      });

      const payload = {
        collection_name: 'test-collection',
        embedding_dimension: 1536,
        metadata_schema: [],
        vdb_endpoint: 'http://localhost:8000',
      };

      result.current.mutate(payload);

      await waitFor(() => expect(result.current.isError).toBe(true));

      // Should fall back to default error message
      expect(result.current.error?.message).toBe('Failed to create collection');
    });

    // NEW TEST: Verify string detail error handling
    it('handles string detail error', async () => {
      fetchSpy.mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: async () => ({
          detail: 'Collection already exists',
        }),
      } as Response);

      const { result } = renderHook(() => useCreateCollection(), {
        wrapper: createWrapper(),
      });

      const payload = {
        collection_name: 'existing-collection',
        embedding_dimension: 1536,
        metadata_schema: [],
        vdb_endpoint: 'http://localhost:8000',
      };

      result.current.mutate(payload);

      await waitFor(() => expect(result.current.isError).toBe(true));

      expect(result.current.error?.message).toBe('Collection already exists');
    });
  });

  describe('useDeleteCollection', () => {
    it('deletes collection successfully', async () => {
      const mockResponse = { 
        status: 'deleted',
        collection_name: 'test-collection'
      };

      fetchSpy.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      } as Response);

      const { result } = renderHook(() => useDeleteCollection(), {
        wrapper: createWrapper(),
      });

      result.current.mutate('test-collection');

      await waitFor(() => expect(result.current.isSuccess).toBe(true));

      expect(fetchSpy).toHaveBeenCalledWith('/api/collections', {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(['test-collection']),
      });
      expect(result.current.data).toEqual(mockResponse);
    });

    it('handles delete error', async () => {
      fetchSpy.mockResolvedValueOnce({
        ok: false,
        status: 404,
        json: async () => ({ message: 'Collection not found' }),
      } as Response);

      const { result } = renderHook(() => useDeleteCollection(), {
        wrapper: createWrapper(),
      });

      result.current.mutate('nonexistent-collection');

      await waitFor(() => expect(result.current.isError).toBe(true));

      expect(result.current.error?.message).toBe('Collection not found');
    });
  });
});


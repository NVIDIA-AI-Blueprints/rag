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
import { useHealthInitialization, useSettingsStore } from '../useSettingsStore';
import type { HealthResponse } from '../../types/api';
import type { ReactNode } from 'react';

// Mock the health API
const mockHealthResponse: HealthResponse = {
  message: 'Service is up.',
  databases: [],
  object_storage: [],
  nim: [
    {
      service: 'LLM',
      url: 'http://llm:8000',
      status: 'healthy',
      latency_ms: 50,
      error: null,
      model: 'meta/llama-3.1-8b-instruct',
      message: null,
      http_status: 200
    },
    {
      service: 'Embeddings', 
      url: 'http://embeddings:8001',
      status: 'healthy',
      latency_ms: 30,
      error: null,
      model: 'nvidia/nv-embedqa-e5-v5',
      message: null,
      http_status: 200
    },
    {
      service: 'Reranker',
      url: 'http://reranker:8002', 
      status: 'healthy',
      latency_ms: 40,
      error: null,
      model: 'nvidia/nv-rerankqa-mistral-4b-v3',
      message: null,
      http_status: 200
    }
  ],
  processing: [],
  task_management: []
};

// Create a wrapper component for React Query
const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        staleTime: Infinity,
      },
    },
  });

  return ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={queryClient}>
      {children}
    </QueryClientProvider>
  );
};

// Mock fetch globally
const mockFetch = vi.fn();
global.fetch = mockFetch;

describe('useHealthInitialization', () => {
  beforeEach(() => {
    // Clear all mocks and reset settings store
    vi.clearAllMocks();
    useSettingsStore.setState({
      model: undefined,
      embeddingModel: undefined,
      rerankerModel: undefined,
      vlmModel: undefined,
      llmEndpoint: undefined,
      embeddingEndpoint: undefined,
      rerankerEndpoint: undefined,
      vlmEndpoint: undefined,
    });

    // Mock console.log to verify initialization logging
    vi.spyOn(console, 'log').mockImplementation(() => {});
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('initializes model settings from health endpoint successfully', async () => {
    // Mock successful health API response
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve(mockHealthResponse),
    });

    const wrapper = createWrapper();
    renderHook(() => useHealthInitialization(), { wrapper });

    // Wait for the health data to be fetched and settings to be updated
    await waitFor(() => {
      const state = useSettingsStore.getState();
      expect(state.model).toBe('meta/llama-3.1-8b-instruct');
      expect(state.embeddingModel).toBe('nvidia/nv-embedqa-e5-v5');
      expect(state.rerankerModel).toBe('nvidia/nv-rerankqa-mistral-4b-v3');
    });

    // Verify endpoints are also set
    await waitFor(() => {
      const state = useSettingsStore.getState();
      expect(state.llmEndpoint).toBe('http://llm:8000');
      expect(state.embeddingEndpoint).toBe('http://embeddings:8001');
      expect(state.rerankerEndpoint).toBe('http://reranker:8002');
    });

    // Verify initialization logging
    expect(console.log).toHaveBeenCalledWith(
      '🔧 Initializing settings from health endpoint:',
      expect.objectContaining({
        model: 'meta/llama-3.1-8b-instruct',
        llmEndpoint: 'http://llm:8000',
        embeddingModel: 'nvidia/nv-embedqa-e5-v5',
        embeddingEndpoint: 'http://embeddings:8001',
        rerankerModel: 'nvidia/nv-rerankqa-mistral-4b-v3',
        rerankerEndpoint: 'http://reranker:8002'
      })
    );
  });

  it('does not override existing user settings', async () => {
    // Pre-populate settings store with user values
    useSettingsStore.setState({
      model: 'user-selected-llm-model',
      embeddingModel: 'user-selected-embedding-model',
    });

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve(mockHealthResponse),
    });

    const wrapper = createWrapper();
    renderHook(() => useHealthInitialization(), { wrapper });

    await waitFor(() => {
      const state = useSettingsStore.getState();
      // Should not override existing user settings
      expect(state.model).toBe('user-selected-llm-model');
      expect(state.embeddingModel).toBe('user-selected-embedding-model');
      // Should still populate undefined fields
      expect(state.rerankerModel).toBe('nvidia/nv-rerankqa-mistral-4b-v3');
    });
  });

  it('handles missing NIM services gracefully', async () => {
    // Mock health response with no NIM services
    const emptyHealthResponse = {
      ...mockHealthResponse,
      nim: []
    };

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve(emptyHealthResponse),
    });

    const wrapper = createWrapper();
    renderHook(() => useHealthInitialization(), { wrapper });

    await waitFor(() => {
      const state = useSettingsStore.getState();
      // Should remain undefined when no services available
      expect(state.model).toBeUndefined();
      expect(state.embeddingModel).toBeUndefined();
      expect(state.rerankerModel).toBeUndefined();
    });

    // Should not log initialization when no updates are made
    expect(console.log).not.toHaveBeenCalledWith(
      '🔧 Initializing settings from health endpoint:',
      expect.anything()
    );
  });

  it('handles health API failure gracefully', async () => {
    // Mock API failure
    mockFetch.mockRejectedValueOnce(new Error('Failed to fetch health status'));

    const wrapper = createWrapper();
    renderHook(() => useHealthInitialization(), { wrapper });

    // Give time for the failed request
    await new Promise(resolve => setTimeout(resolve, 100));

    const state = useSettingsStore.getState();
    // Should remain undefined when API fails
    expect(state.model).toBeUndefined();
    expect(state.embeddingModel).toBeUndefined();
    expect(state.rerankerModel).toBeUndefined();
  });

  it('handles services with missing model information', async () => {
    // Mock health response with services but no model field
    const healthResponseNoModels = {
      ...mockHealthResponse,
      nim: [
        {
          service: 'LLM',
          url: 'http://llm:8000',
          status: 'healthy',
          latency_ms: 50,
          error: null,
          model: null, // No model information
          message: null,
          http_status: 200
        }
      ]
    };

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve(healthResponseNoModels),
    });

    const wrapper = createWrapper();
    renderHook(() => useHealthInitialization(), { wrapper });

    await waitFor(() => {
      const state = useSettingsStore.getState();
      // Should not set model when it's null/undefined in health response
      expect(state.model).toBeUndefined();
    });
  });

  it('maps service names correctly using case-insensitive matching', async () => {
    // Test various service name variations
    const variousServiceNames = {
      ...mockHealthResponse,
      nim: [
        {
          service: 'llm-service', // lowercase with hyphen
          url: 'http://llm:8000',
          status: 'healthy',
          latency_ms: 50,
          error: null,
          model: 'test-llm-model',
          message: null,
          http_status: 200
        },
        {
          service: 'EMBEDDING_SERVICE', // uppercase with underscore  
          url: 'http://embed:8001',
          status: 'healthy',
          latency_ms: 30,
          error: null,
          model: 'test-embedding-model',
          message: null,
          http_status: 200
        }
      ]
    };

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve(variousServiceNames),
    });

    const wrapper = createWrapper();
    renderHook(() => useHealthInitialization(), { wrapper });

    await waitFor(() => {
      const state = useSettingsStore.getState();
      expect(state.model).toBe('test-llm-model');
      expect(state.embeddingModel).toBe('test-embedding-model');
    });
  });

  it('only initializes settings when health data is available and not loading', async () => {
    // Mock loading state (no data yet)
    mockFetch.mockImplementation(() => new Promise(() => {})); // Never resolves

    const wrapper = createWrapper();
    renderHook(() => useHealthInitialization(), { wrapper });

    // Should not initialize while loading
    const state = useSettingsStore.getState();
    expect(state.model).toBeUndefined();
    expect(console.log).not.toHaveBeenCalled();
  });
});

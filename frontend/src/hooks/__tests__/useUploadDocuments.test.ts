// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useUploadDocuments } from '../useUploadDocuments';

// Mock the notification store
const mockAddTaskNotification = vi.fn();
vi.mock('../../store/useNotificationStore', () => ({
  useNotificationStore: () => ({
    addTaskNotification: mockAddTaskNotification
  })
}));

// Mock the toast store
const mockShowToast = vi.fn();
vi.mock('../../store/useToastStore', () => ({
  useToastStore: () => ({
    showToast: mockShowToast
  })
}));

// Mock fetch globally
global.fetch = vi.fn();

describe('useUploadDocuments', () => {
  const mockFetch = global.fetch as ReturnType<typeof vi.fn>;

  beforeEach(() => {
    vi.clearAllMocks();
    mockFetch.mockClear();
  });

  it('should initialize with isPending false', () => {
    const { result } = renderHook(() => useUploadDocuments());
    
    expect(result.current.isPending).toBe(false);
    expect(typeof result.current.mutate).toBe('function');
  });

  it('should set isPending to true during upload and false after success', async () => {
    const mockResponse = {
      ok: true,
      json: () => Promise.resolve({ 
        collection_name: 'test-collection',
        task_id: 'task-123'
      })
    };
    mockFetch.mockResolvedValueOnce(mockResponse);

    const { result } = renderHook(() => useUploadDocuments());
    const onSuccess = vi.fn();
    
    // Initial state
    expect(result.current.isPending).toBe(false);
    
    // Start upload
    act(() => {
      result.current.mutate(
        { 
          files: [new File(['content'], 'test.txt', { type: 'text/plain' })], 
          metadata: { collection_name: 'test-collection' } 
        },
        { onSuccess }
      );
    });
    
    // Should be pending immediately
    expect(result.current.isPending).toBe(true);
    
    // Wait for async operation to complete
    await act(async () => {
      await new Promise(resolve => setTimeout(resolve, 0));
    });
    
    // Should no longer be pending
    expect(result.current.isPending).toBe(false);
    expect(onSuccess).toHaveBeenCalledWith({
      collection_name: 'test-collection',
      task_id: 'task-123'
    });
  });

  it('should set isPending to false after error', async () => {
    const mockError = new Error('Upload failed');
    mockFetch.mockRejectedValueOnce(mockError);

    const { result } = renderHook(() => useUploadDocuments());
    const onError = vi.fn();

    // Start upload
    act(() => {
      result.current.mutate(
        {
          files: [new File(['content'], 'test.txt', { type: 'text/plain' })],
          metadata: { collection_name: 'test-collection' }
        },
        { onError }
      );
    });

    // Should be pending
    expect(result.current.isPending).toBe(true);

    // Wait for async operation to complete
    await act(async () => {
      await new Promise(resolve => setTimeout(resolve, 0));
    });

    // Should no longer be pending after error
    expect(result.current.isPending).toBe(false);
    expect(onError).toHaveBeenCalledWith(mockError);
    expect(mockShowToast).toHaveBeenCalledWith('Upload failed', 'error');
  });

  it('surfaces backend message field as toast on HTTP 503 (e.g., Elasticsearch unavailable)', async () => {
    const mockResponse = {
      ok: false,
      status: 503,
      json: () => Promise.resolve({
        message: 'Vector database (Elasticsearch) is unavailable at http://elasticsearch:9200. Please verify Elasticsearch is running and accessible.'
      })
    };
    mockFetch.mockResolvedValueOnce(mockResponse);

    const { result } = renderHook(() => useUploadDocuments());
    const onError = vi.fn();

    act(() => {
      result.current.mutate(
        {
          files: [new File(['content'], 'test.txt', { type: 'text/plain' })],
          metadata: { collection_name: 'test-collection' }
        },
        { onError }
      );
    });

    await act(async () => {
      await new Promise(resolve => setTimeout(resolve, 0));
    });

    expect(mockShowToast).toHaveBeenCalledWith(
      'Vector database (Elasticsearch) is unavailable at http://elasticsearch:9200. Please verify Elasticsearch is running and accessible.',
      'error'
    );
    expect(onError).toHaveBeenCalledTimes(1);
    expect((onError.mock.calls[0][0] as Error).message).toBe(
      'Vector database (Elasticsearch) is unavailable at http://elasticsearch:9200. Please verify Elasticsearch is running and accessible.'
    );
  });

  it('surfaces backend message field as toast on HTTP 500 (e.g., missing collection)', async () => {
    const mockResponse = {
      ok: false,
      status: 500,
      json: () => Promise.resolve({
        message: 'Ingestion of files failed with error: Collection test_collection does not exist. Ensure a collection is created using POST /collection endpoint first.'
      })
    };
    mockFetch.mockResolvedValueOnce(mockResponse);

    const { result } = renderHook(() => useUploadDocuments());
    const onError = vi.fn();

    act(() => {
      result.current.mutate(
        {
          files: [new File(['content'], 'test.txt')],
          metadata: { collection_name: 'test_collection' }
        },
        { onError }
      );
    });

    await act(async () => {
      await new Promise(resolve => setTimeout(resolve, 0));
    });

    expect(mockShowToast).toHaveBeenCalledWith(
      expect.stringContaining('Collection test_collection does not exist'),
      'error'
    );
  });

  it('extracts Pydantic detail[0].msg and strips "Value error, " prefix on HTTP 422', async () => {
    const mockResponse = {
      ok: false,
      status: 422,
      json: () => Promise.resolve({
        detail: [{ msg: "Value error, Collection name must be lowercase", type: 'value_error', loc: ['body'] }]
      })
    };
    mockFetch.mockResolvedValueOnce(mockResponse);

    const { result } = renderHook(() => useUploadDocuments());
    const onError = vi.fn();

    act(() => {
      result.current.mutate(
        {
          files: [new File(['content'], 'test.txt')],
          metadata: { collection_name: 'BadName' }
        },
        { onError }
      );
    });

    await act(async () => {
      await new Promise(resolve => setTimeout(resolve, 0));
    });

    expect(mockShowToast).toHaveBeenCalledWith('Collection name must be lowercase', 'error');
  });

  it('falls back to default message when error response body cannot be parsed', async () => {
    const mockResponse = {
      ok: false,
      status: 500,
      json: () => Promise.reject(new Error('Not JSON'))
    };
    mockFetch.mockResolvedValueOnce(mockResponse);

    const { result } = renderHook(() => useUploadDocuments());
    const onError = vi.fn();

    act(() => {
      result.current.mutate(
        {
          files: [new File(['content'], 'test.txt')],
          metadata: { collection_name: 'c' }
        },
        { onError }
      );
    });

    await act(async () => {
      await new Promise(resolve => setTimeout(resolve, 0));
    });

    expect(mockShowToast).toHaveBeenCalledWith('Failed to upload documents', 'error');
  });

  it('should handle multiple concurrent uploads correctly', async () => {
    const mockResponse1 = {
      ok: true,
      json: () => Promise.resolve({ collection_name: 'test-1', task_id: 'task-1' })
    };
    const mockResponse2 = {
      ok: true,
      json: () => Promise.resolve({ collection_name: 'test-2', task_id: 'task-2' })
    };
    
    mockFetch
      .mockResolvedValueOnce(mockResponse1)
      .mockResolvedValueOnce(mockResponse2);

    const { result } = renderHook(() => useUploadDocuments());
    const onSuccess1 = vi.fn();
    const onSuccess2 = vi.fn();
    
    // Start first upload
    act(() => {
      result.current.mutate(
        { files: [new File(['1'], 'test1.txt')], metadata: { collection_name: 'test-1' } },
        { onSuccess: onSuccess1 }
      );
    });
    
    expect(result.current.isPending).toBe(true);
    
    // Start second upload while first is still pending
    act(() => {
      result.current.mutate(
        { files: [new File(['2'], 'test2.txt')], metadata: { collection_name: 'test-2' } },
        { onSuccess: onSuccess2 }
      );
    });
    
    // Should still be pending
    expect(result.current.isPending).toBe(true);
    
    // Wait for both uploads to complete
    await act(async () => {
      await new Promise(resolve => setTimeout(resolve, 0));
    });
    
    // Should no longer be pending
    expect(result.current.isPending).toBe(false);
  });

  it('should create proper FormData with files and metadata', async () => {
    const mockResponse = {
      ok: true,
      json: () => Promise.resolve({ collection_name: 'test-collection', task_id: 'task-123' })
    };
    mockFetch.mockResolvedValueOnce(mockResponse);

    const { result } = renderHook(() => useUploadDocuments());
    
    const testFile = new File(['test content'], 'test.txt', { type: 'text/plain' });
    const metadata = { 
      collection_name: 'test-collection',
      custom_field: 'custom_value'
    };
    
    act(() => {
      result.current.mutate({ files: [testFile], metadata }, { onSuccess: vi.fn() });
    });
    
    await act(async () => {
      await new Promise(resolve => setTimeout(resolve, 0));
    });
    
    expect(mockFetch).toHaveBeenCalledWith(
      '/api/documents?blocking=false',
      expect.objectContaining({
        method: 'POST',
        body: expect.any(FormData)
      })
    );
    
    // Verify FormData structure
    const callArgs = mockFetch.mock.calls[0];
    const formData = callArgs[1].body as FormData;
    
    expect(formData.get('documents')).toBe(testFile);
    expect(formData.get('data')).toBe(JSON.stringify({ ...metadata, generate_summary: true }));
  });

  it('should add task notification on successful upload', async () => {
    const mockResponse = {
      ok: true,
      json: () => Promise.resolve({ 
        collection_name: 'test-collection',
        task_id: 'task-123'
      })
    };
    mockFetch.mockResolvedValueOnce(mockResponse);

    const { result } = renderHook(() => useUploadDocuments());
    
    act(() => {
      result.current.mutate(
        { 
          files: [new File(['content'], 'test.txt')], 
          metadata: { collection_name: 'test-collection' } 
        },
        { onSuccess: vi.fn() }
      );
    });
    
    await act(async () => {
      await new Promise(resolve => setTimeout(resolve, 0));
    });
    
    expect(mockAddTaskNotification).toHaveBeenCalledWith({
      id: 'task-123',
      collection_name: 'test-collection',
      documents: ['test.txt'],
      created_at: expect.any(String),
      state: 'PENDING'
    });
  });
});

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
    expect(formData.get('data')).toBe(JSON.stringify(metadata));
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

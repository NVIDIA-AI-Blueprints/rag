/**
 * Test data factories for API mocks.
 */
import type {
  ConfigurationResponse,
  HealthResponse,
  CollectionDocumentsResponse,
  IngestionTask,
} from '../../src/types/api.ts';

export interface MockCollection {
  collection_name: string;
  num_entities?: number;
  metadata_schema?: Array<Record<string, unknown>>;
  [key: string]: unknown;
}

export const buildHealthy = (): HealthResponse => ({
  message: 'Service is up.',
  databases: [
    {
      service: 'milvus',
      url: 'http://milvus:19530',
      status: 'healthy',
      latency_ms: 1.2,
      error: null,
      collections: { count: 1 },
    },
  ],
  object_storage: [
    {
      service: 'minio',
      url: 'http://minio:9000',
      status: 'healthy',
      latency_ms: 2.4,
      error: null,
      buckets: 1,
      message: null,
    },
  ],
  nim: [
    {
      service: 'embedding',
      url: 'http://embedding:8000',
      status: 'healthy',
      latency_ms: 3.5,
      error: null,
      model: 'nvidia/nv-embedqa',
      message: null,
      http_status: 200,
    },
  ],
  processing: [
    {
      service: 'nv-ingest',
      url: 'http://nv-ingest:7670',
      status: 'healthy',
      latency_ms: 2.0,
      error: null,
      http_status: 200,
    },
  ],
  task_management: [
    {
      service: 'redis',
      url: 'redis://redis:6379',
      status: 'healthy',
      latency_ms: 0.8,
      error: null,
      message: null,
    },
  ],
});

export const buildUnhealthy = (): HealthResponse => {
  const h = buildHealthy();
  h.databases[0].status = 'unhealthy';
  h.databases[0].error = 'connection refused';
  return h;
};

export const buildCollections = (
  names: string[] = ['docs', 'reports'],
): { collections: MockCollection[] } => ({
  collections: names.map((name) => ({
    collection_name: name,
    num_entities: 42,
    metadata_schema: [],
  })),
});

export const buildCollectionDocuments = (
  documents: Array<{ name: string; description?: string; tags?: string[] }> = [],
): CollectionDocumentsResponse => ({
  message: 'ok',
  total_documents: documents.length,
  documents: documents.map((d) => ({
    document_name: d.name,
    metadata: {},
    document_info: {
      description: d.description,
      tags: d.tags,
      document_type: 'pdf',
      file_size: 1024,
      date_created: new Date().toISOString(),
      total_elements: 10,
    },
  })),
});

export const buildConfiguration = (): ConfigurationResponse => ({
  rag_configuration: {
    temperature: 0.2,
    top_p: 0.7,
    max_tokens: 1024,
    vdb_top_k: 20,
    reranker_top_k: 4,
    confidence_threshold: 0.3,
  },
  feature_toggles: {
    enable_reranker: true,
    enable_citations: true,
    enable_guardrails: false,
    enable_query_rewriting: true,
    enable_vlm_inference: false,
    enable_filter_generator: false,
  },
  models: {
    llm_model: 'meta/llama-3.1-8b-instruct',
    embedding_model: 'nvidia/nv-embedqa-e5-v5',
    reranker_model: 'nvidia/llama-3.2-nv-rerankqa-1b-v2',
    vlm_model: 'meta/llama-3.2-11b-vision-instruct',
  },
  endpoints: {
    llm_endpoint: 'http://llm:8000/v1/chat/completions',
    embedding_endpoint: 'http://embedding:8000/v1/embeddings',
    reranker_endpoint: 'http://reranker:8000/v1/ranking',
    vlm_endpoint: 'http://vlm:8000/v1/chat/completions',
    vdb_endpoint: 'http://milvus:19530',
  },
});

export const buildTaskPending = (id = 'task-1', collection = 'docs'): IngestionTask => ({
  id,
  collection_name: collection,
  created_at: new Date().toISOString(),
  state: 'PENDING',
  documents: ['doc1.pdf'],
  result: {
    message: 'in progress',
    total_documents: 1,
    documents: [],
    failed_documents: [],
    documents_completed: 0,
    batches_completed: 0,
  },
});

export const buildTaskFinished = (id = 'task-1', collection = 'docs'): IngestionTask => ({
  id,
  collection_name: collection,
  created_at: new Date().toISOString(),
  state: 'FINISHED',
  documents: ['doc1.pdf'],
  result: {
    message: 'done',
    total_documents: 1,
    documents: [{ document_id: 'd1', document_name: 'doc1.pdf', size_bytes: 1024 }],
    failed_documents: [],
  },
});

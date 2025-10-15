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

import type { APIMetadataField } from "./collections";

/**
 * Payload structure for creating a new collection.
 */
export interface CreateCollectionPayload {
  collection_name: string;
  embedding_dimension: number;
  metadata_schema: APIMetadataField[];
  vdb_endpoint?: string;
}

/**
 * Represents a document item within a collection.
 */
export interface DocumentItem {
  document_name: string;
  metadata: Record<string, string>;
}

/**
 * Response structure for fetching collection documents.
 */
export interface CollectionDocumentsResponse {
  message: string;
  total_documents: number;
  documents: DocumentItem[];
}

export interface IngestionTask {
  id: string; // You'll likely assign this manually, since it's not in the /status response
  collection_name: string; // Also tracked locally, not in the response
  created_at: string;

  state: "PENDING" | "FINISHED" | "FAILED" | "UNKNOWN";

  documents?: string[];

  result?: {
    message: string;
    total_documents: number;
    documents: {
      document_id: string;
      document_name: string;
      size_bytes?: number;
    }[];
    failed_documents: {
      document_name: string;
      error_message?: string;
    }[];
    validation_errors?: unknown[]; // Fill in if needed
  };
}

/**
 * Health check response from the ingestor server.
 */
export interface HealthResponse {
  message: string;
  databases: Array<{
    service: string;
    url: string;
    status: string;
    latency_ms: number;
    error: string | null;
    collections: unknown;
  }>;
  object_storage: Array<{
    service: string;
    url: string;
    status: string;
    latency_ms: number;
    error: string | null;
    buckets: number;
    message: string | null;
  }>;
  nim: Array<{
    service: string;
    url: string;
    status: string;
    latency_ms: number;
    error: string | null;
    model: string;
    message: string | null;
    http_status: number | null;
  }>;
  processing: Array<{
    service: string;
    url: string;
    status: string;
    latency_ms: number;
    error: string | null;
    http_status: number;
  }>;
  task_management: Array<{
    service: string;
    url: string;
    status: string;
    latency_ms: number;
    error: string | null;
    message: string | null;
  }>;
}
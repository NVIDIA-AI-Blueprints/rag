# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Shared Pydantic models for health check responses across all servers.

These models provide type safety and consistency for health check responses
in both the RAG server and Ingestor server.
"""

from enum import Enum

from pydantic import BaseModel, Field


class ServiceStatus(str, Enum):
    """Enum for service health status values."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"
    ERROR = "error"
    UNKNOWN = "unknown"


class BaseServiceHealthInfo(BaseModel):
    """Base health info model with common fields for all services"""

    service: str
    url: str
    status: str | ServiceStatus
    latency_ms: float = 0
    error: str | None = None


class DatabaseHealthInfo(BaseServiceHealthInfo):
    """Health info specific to database services"""

    collections: int | None = None


class StorageHealthInfo(BaseServiceHealthInfo):
    """Health info specific to object storage services"""

    buckets: int | None = None
    message: str | None = None


class NIMServiceHealthInfo(BaseServiceHealthInfo):
    """Health info specific to NIM services (LLM, embeddings, etc.)"""

    model: str | None = None
    message: str | None = None
    http_status: int | None = None


class ProcessingHealthInfo(BaseServiceHealthInfo):
    """Health info specific to document processing services"""

    http_status: int | None = None


class TaskManagementHealthInfo(BaseServiceHealthInfo):
    """Health info specific to task management services"""

    message: str | None = None


class HealthResponseBase(BaseModel):
    """Base health response with common fields for all servers"""

    message: str = Field(max_length=4096, pattern=r"[\s\S]*", default="Service is up.")
    databases: list[DatabaseHealthInfo] = Field(default_factory=list)
    object_storage: list[StorageHealthInfo] = Field(default_factory=list)
    nim: list[NIMServiceHealthInfo] = Field(
        default_factory=list
    )  # NIM services (embeddings, LLM, etc.)


class RAGHealthResponse(HealthResponseBase):
    """Health response for RAG server with database, storage, and NIM services"""

    pass


class IngestorHealthResponse(HealthResponseBase):
    """Health response for Ingestor server with additional processing and task management services"""

    processing: list[ProcessingHealthInfo] = Field(
        default_factory=list
    )  # Document processing services (e.g., NV-Ingest)
    task_management: list[TaskManagementHealthInfo] = Field(
        default_factory=list
    )  # Task management services (e.g., Redis)


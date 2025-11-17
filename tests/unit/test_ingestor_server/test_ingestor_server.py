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

import asyncio
import json
from io import BytesIO
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from nvidia_rag.ingestor_server.server import (
    DocumentUploadRequest,
    SummaryOptions,
)


class MockNvidiaRAGIngestor:
    """Mock class for NvidiaRAGIngestor with configurable responses and error states"""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset to default state"""
        self._upload_side_effect = None
        self._status_side_effect = None
        self._get_documents_side_effect = None
        self._get_collections_side_effect = None
        self._create_collection_side_effect = None
        self._delete_collections_side_effect = None
        self._delete_documents_side_effect = None
        self._health_side_effect = None
        self._validate_directory_traversal_attack_side_effect = None

    async def health(self, check_dependencies: bool = False):
        """Mock health method"""
        if self._health_side_effect:
            return self._health_side_effect(check_dependencies)

        # Default response for basic health check
        response = {"message": "Service is up."}

        if check_dependencies:
            # Mock dependencies health results
            response.update(
                {
                    "databases": [
                        {
                            "service": "Milvus",
                            "url": "http://localhost:19530",
                            "status": "healthy",
                            "latency_ms": 15.5,
                            "collections": 3,
                            "error": None,
                        }
                    ],
                    "object_storage": [
                        {
                            "service": "MinIO",
                            "url": "http://localhost:9000",
                            "status": "healthy",
                            "latency_ms": 8.2,
                            "buckets": 2,
                            "error": None,
                        }
                    ],
                    "nim": [
                        {
                            "service": "Embeddings (nvidia/nv-embedqa-e5-v5)",
                            "model": "nvidia/nv-embedqa-e5-v5",
                            "url": "NVIDIA API Catalog",
                            "status": "healthy",
                            "latency_ms": 0,
                            "message": "Using NVIDIA API Catalog",
                            "error": None,
                        }
                    ],
                    "processing": [
                        {
                            "service": "NV-Ingest",
                            "url": "localhost:7670",
                            "status": "healthy",
                            "latency_ms": 25.3,
                            "http_status": 200,
                            "error": None,
                        }
                    ],
                    "task_management": [
                        {
                            "service": "Redis",
                            "url": "localhost:6379",
                            "status": "healthy",
                            "latency_ms": 5.1,
                            "error": None,
                        }
                    ],
                }
            )

        return response

    async def upload_documents(self, *args, **kwargs):
        """Mock upload_documents method"""
        if self._upload_side_effect:
            return self._upload_side_effect(*args, **kwargs)
        return {
            "message": "Document upload job successfully completed.",
            "total_documents": 1,
            "documents": [
                {
                    "document_id": "test-doc-id",
                    "document_name": "test.txt",
                    "size_bytes": 1024,
                    "metadata": {},
                }
            ],
            "failed_documents": [],
            "validation_errors": [],
        }

    async def status(self, task_id: str):
        """Mock status method"""
        if self._status_side_effect:
            return self._status_side_effect(task_id)
        return {
            "state": "FINISHED",
            "result": {
                "message": "Document upload job successfully completed.",
                "total_documents": 1,
                "documents": [
                    {
                        "document_id": "test-doc-id",
                        "document_name": "test.txt",
                        "size_bytes": 1024,
                        "metadata": {},
                    }
                ],
                "failed_documents": [],
                "validation_errors": [],
            },
        }

    def get_documents(self, collection_name: str, vdb_endpoint: str, vdb_auth_token: str = ""):
        """Mock get_documents method"""
        if self._get_documents_side_effect:
            return self._get_documents_side_effect(collection_name, vdb_endpoint, vdb_auth_token)
        return {
            "documents": [
                {
                    "document_id": "",
                    "document_name": "test.txt",
                    "timestamp": "",
                    "size_bytes": 0,
                    "metadata": {},
                }
            ],
            "total_documents": 1,
            "message": "Document listing successfully completed.",
        }

    def get_collections(self, vdb_endpoint: str, vdb_auth_token: str = ""):
        """Mock get_collections method"""
        if self._get_collections_side_effect:
            return self._get_collections_side_effect(vdb_endpoint, vdb_auth_token)
        return {
            "collections": [
                {
                    "collection_name": "test_collection",
                    "description": "Test collection",
                    "document_count": 1,
                }
            ],
            "total_collections": 1,
            "message": "Collections listed successfully.",
        }

    def create_collection(
        self,
        collection_name: str,
        vdb_endpoint: str,
        embedding_dimension: int = 2048,
        metadata_schema: list = None,
        description: str = "",
        tags: list = None,
        owner: str = "",
        created_by: str = "",
        business_domain: str = "",
        status: str = "Active",
        vdb_auth_token: str = "",
    ):
        """Mock create_collection method"""
        if self._create_collection_side_effect:
            return self._create_collection_side_effect(
                collection_name,
                vdb_endpoint,
                embedding_dimension,
                metadata_schema,
                description,
                tags,
                owner,
                created_by,
                business_domain,
                status,
                vdb_auth_token
            )
        return {
            "message": f"Collection {collection_name} created successfully.",
            "collection_name": collection_name,
        }

    def delete_collections(self, vdb_endpoint: str, collection_names: list, vdb_auth_token: str = ""):
        """Mock delete_collections method"""
        if self._delete_collections_side_effect:
            return self._delete_collections_side_effect(vdb_endpoint, collection_names, vdb_auth_token)
        # Filter out None values and ensure all items are strings
        valid_collections = [str(name) for name in collection_names if name is not None]
        return {
            "message": "Collections deleted successfully",
            "successful": valid_collections,
            "failed": [],
            "total_successful": len(valid_collections),
            "total_failed": 0,
        }

    def delete_documents(
        self,
        document_names: list,
        collection_name: str,
        vdb_endpoint: str,
        include_upload_path: bool = False,
        vdb_auth_token: str = "",
    ):
        """Mock delete_documents method"""
        if self._delete_documents_side_effect:
            return self._delete_documents_side_effect(
                document_names, collection_name, vdb_endpoint, include_upload_path, vdb_auth_token
            )
        return {
            "message": "Files deleted successfully",
            "total_documents": len(document_names),
            "documents": [
                {"document_id": "", "document_name": doc, "size_bytes": 0}
                for doc in document_names
            ],
        }

    # Error methods
    def raise_upload_error(self):
        def error(*args, **kwargs):
            raise Exception("Upload failed")

        self._upload_side_effect = error

    def raise_upload_validation_error(self):
        def error(*args, **kwargs):
            raise ValueError("Invalid upload data")

        self._upload_side_effect = error

    def raise_status_error(self):
        def error(task_id):
            return {"state": "FAILED", "result": {"message": "Status check failed"}}

        self._status_side_effect = error

    def return_status_not_found(self):
        def not_found(task_id):
            return {"state": "UNKNOWN", "result": {"message": "Task not found"}}

        self._status_side_effect = not_found

    def return_empty_documents(self):
        def empty(collection_name, vdb_endpoint, vdb_auth_token=""):
            return {
                "documents": [],
                "total_documents": 0,
                "message": "No documents found",
            }

        self._get_documents_side_effect = empty

    def raise_get_documents_error(self):
        def error(collection_name, vdb_endpoint, vdb_auth_token=""):
            raise Exception("Failed to get documents")

        self._get_documents_side_effect = error

    def return_empty_collections(self):
        def empty(vdb_endpoint, vdb_auth_token=""):
            return {
                "collections": [],
                "total_collections": 0,
                "message": "No collections found",
            }

        self._get_collections_side_effect = empty

    def raise_get_collections_error(self):
        def error(vdb_endpoint, vdb_auth_token=""):
            raise Exception("Failed to get collections")

        self._get_collections_side_effect = error

    def raise_create_collection_error(self):
        def error(collection_name, vdb_endpoint, embedding_dimension, metadata_schema, vdb_auth_token=""):
            raise Exception("Failed to create collection")

        self._create_collection_side_effect = error

    def raise_delete_collections_error(self):
        def error(vdb_endpoint, collection_names, vdb_auth_token=""):
            raise Exception("Failed to delete collections")

        self._delete_collections_side_effect = error

    def raise_delete_documents_error(self):
        def error(document_names, collection_name, vdb_endpoint, include_upload_path, vdb_auth_token=""):
            raise Exception("Failed to delete documents")

        self._delete_documents_side_effect = error

    async def validate_directory_traversal_attack(self, file):
        if self._validate_directory_traversal_attack_side_effect:
            return self._validate_directory_traversal_attack_side_effect(file)


# Create mock instances
mock_nvidia_rag_ingestor_instance = MockNvidiaRAGIngestor()


# Common fixtures
@pytest.fixture(scope="module")
def setup_test_env():
    """Setup test environment with all necessary mocks"""
    with patch(
        "nvidia_rag.ingestor_server.server.NV_INGEST_INGESTOR",
        mock_nvidia_rag_ingestor_instance,
    ):
        from nvidia_rag.ingestor_server.server import app

        yield app


@pytest.fixture
def client(setup_test_env):
    """Create test client"""
    return TestClient(setup_test_env)


@pytest.fixture(autouse=True)
def reset_mock_instance():
    """Reset mock instance before each test"""
    mock_nvidia_rag_ingestor_instance.reset()
    yield


@pytest.fixture
def mock_ingestor():
    """Provide access to the mock ingestor instance"""
    return mock_nvidia_rag_ingestor_instance


class TestHealthEndpoint:
    """Tests for the /health endpoint"""

    def test_health_check_basic(self, client):
        """Test basic health check without dependencies"""
        response = client.get("/v1/health")
        assert response.status_code == 200

        response_data = response.json()
        assert "message" in response_data
        assert response_data["message"] == "Service is up."

        # Basic health check should not include dependency data
        assert "databases" in response_data
        assert "object_storage" in response_data
        assert "nim" in response_data
        assert "processing" in response_data
        assert "task_management" in response_data

        # All dependency arrays should be empty for basic health check
        assert response_data["databases"] == []
        assert response_data["object_storage"] == []
        assert response_data["nim"] == []
        assert response_data["processing"] == []
        assert response_data["task_management"] == []

    def test_health_check_with_dependencies(self, client):
        """Test health check with dependencies enabled"""
        response = client.get("/v1/health?check_dependencies=true")
        assert response.status_code == 200

        response_data = response.json()
        assert "message" in response_data
        assert response_data["message"] == "Service is up."

        # Should include dependency health data
        assert "databases" in response_data
        assert "object_storage" in response_data
        assert "nim" in response_data
        assert "processing" in response_data
        assert "task_management" in response_data

        # Verify database health info structure
        assert len(response_data["databases"]) == 1
        db_health = response_data["databases"][0]
        assert db_health["service"] == "Milvus"
        assert db_health["url"] == "http://localhost:19530"
        assert db_health["status"] == "healthy"
        assert "latency_ms" in db_health
        assert "collections" in db_health
        assert db_health["error"] is None

        # Verify object storage health info structure
        assert len(response_data["object_storage"]) == 1
        storage_health = response_data["object_storage"][0]
        assert storage_health["service"] == "MinIO"
        assert storage_health["url"] == "http://localhost:9000"
        assert storage_health["status"] == "healthy"
        assert "latency_ms" in storage_health
        assert "buckets" in storage_health

        # Verify NIM service health info structure
        assert len(response_data["nim"]) == 1
        nim_health = response_data["nim"][0]
        assert "Embeddings" in nim_health["service"]
        assert nim_health["url"] == "NVIDIA API Catalog"
        assert nim_health["status"] == "healthy"
        assert "message" in nim_health
        assert "model" in nim_health
        assert nim_health["model"] == "nvidia/nv-embedqa-e5-v5"

        # Verify processing service health info structure
        assert len(response_data["processing"]) == 1
        processing_health = response_data["processing"][0]
        assert processing_health["service"] == "NV-Ingest"
        assert processing_health["url"] == "localhost:7670"
        assert processing_health["status"] == "healthy"
        assert "http_status" in processing_health

        # Verify task management service health info structure
        assert len(response_data["task_management"]) == 1
        task_health = response_data["task_management"][0]
        assert task_health["service"] == "Redis"
        assert task_health["url"] == "localhost:6379"
        assert task_health["status"] == "healthy"

    def test_health_check_dependencies_false_explicit(self, client):
        """Test health check with dependencies explicitly set to false"""
        response = client.get("/v1/health?check_dependencies=false")
        assert response.status_code == 200

        response_data = response.json()
        assert response_data["message"] == "Service is up."

        # Should not include detailed dependency data
        assert response_data["databases"] == []
        assert response_data["object_storage"] == []
        assert response_data["nim"] == []
        assert response_data["processing"] == []
        assert response_data["task_management"] == []

    def test_health_check_with_unhealthy_dependencies(self, client, mock_ingestor):
        """Test health check when some dependencies are unhealthy"""

        # Configure mock to return unhealthy dependencies
        def unhealthy_health_response(check_dependencies: bool = False):
            response = {"message": "Service is up."}
            if check_dependencies:
                response.update(
                    {
                        "databases": [
                            {
                                "service": "Milvus",
                                "url": "http://localhost:19530",
                                "status": "error",
                                "latency_ms": 0,
                                "collections": None,
                                "error": "Connection refused",
                            }
                        ],
                        "object_storage": [
                            {
                                "service": "MinIO",
                                "url": "http://localhost:9000",
                                "status": "timeout",
                                "latency_ms": 5000,
                                "buckets": None,
                                "error": "Request timed out after 5s",
                            }
                        ],
                        "nim": [
                            {
                                "service": "Embeddings",
                                "model": "nvidia/nv-embedqa-e5-v5",
                                "url": "NVIDIA API Catalog",
                                "status": "error",
                                "latency_ms": 0,
                                "error": "API rate limit exceeded",
                            }
                        ],
                        "processing": [],
                        "task_management": [],
                    }
                )
            return response

        mock_ingestor._health_side_effect = unhealthy_health_response

        response = client.get("/v1/health?check_dependencies=true")
        assert response.status_code == 200  # Health endpoint should still return 200

        response_data = response.json()
        assert response_data["message"] == "Service is up."

        # Verify unhealthy database status
        assert len(response_data["databases"]) == 1
        db_health = response_data["databases"][0]
        assert db_health["status"] == "error"
        assert db_health["error"] == "Connection refused"

        # Verify timeout storage status
        assert len(response_data["object_storage"]) == 1
        storage_health = response_data["object_storage"][0]
        assert storage_health["status"] == "timeout"
        assert storage_health["error"] == "Request timed out after 5s"

    def test_health_check_with_skipped_services(self, client, mock_ingestor):
        """Test health check when some services are skipped"""

        def skipped_services_response(check_dependencies: bool = False):
            response = {"message": "Service is up."}
            if check_dependencies:
                response.update(
                    {
                        "databases": [
                            {
                                "service": "Milvus",
                                "url": "",
                                "status": "skipped",
                                "latency_ms": 0,
                                "collections": None,
                                "error": "No URL provided",
                            }
                        ],
                        "object_storage": [],
                        "nim": [
                            {
                                "service": "Embeddings",
                                "model": "nvidia/nv-embedqa-e5-v5",
                                "url": "",
                                "status": "skipped",
                                "latency_ms": 0,
                                "error": "No URL provided",
                            }
                        ],
                        "processing": [],
                        "task_management": [],
                    }
                )
            return response

        mock_ingestor._health_side_effect = skipped_services_response

        response = client.get("/v1/health?check_dependencies=true")
        assert response.status_code == 200

        response_data = response.json()
        db_health = response_data["databases"][0]
        assert db_health["status"] == "skipped"
        assert db_health["error"] == "No URL provided"

    def test_health_check_with_mixed_service_statuses(self, client, mock_ingestor):
        """Test health check with a mix of healthy and unhealthy services"""

        def mixed_health_response(check_dependencies: bool = False):
            response = {"message": "Service is up."}
            if check_dependencies:
                response.update(
                    {
                        "databases": [
                            {
                                "service": "Milvus",
                                "url": "http://localhost:19530",
                                "status": "healthy",
                                "latency_ms": 15.5,
                                "collections": 3,
                                "error": None,
                            }
                        ],
                        "object_storage": [
                            {
                                "service": "MinIO",
                                "url": "http://localhost:9000",
                                "status": "error",
                                "latency_ms": 0,
                                "buckets": None,
                                "error": "Connection refused",
                            }
                        ],
                        "nim": [
                            {
                                "service": "Embeddings (nvidia/nv-embedqa-e5-v5)",
                                "model": "nvidia/nv-embedqa-e5-v5",
                                "url": "NVIDIA API Catalog",
                                "status": "healthy",
                                "latency_ms": 0,
                                "message": "Using NVIDIA API Catalog",
                                "error": None,
                            }
                        ],
                        "processing": [],
                        "task_management": [],
                    }
                )
            return response

        mock_ingestor._health_side_effect = mixed_health_response

        response = client.get("/v1/health?check_dependencies=true")
        assert (
            response.status_code == 200
        )  # Service is still up even if some dependencies are down

        response_data = response.json()
        assert response_data["message"] == "Service is up."

        # Verify we have both healthy and unhealthy services
        assert len(response_data["databases"]) == 1
        assert response_data["databases"][0]["status"] == "healthy"

        assert len(response_data["object_storage"]) == 1
        assert response_data["object_storage"][0]["status"] == "error"
        assert response_data["object_storage"][0]["error"] == "Connection refused"

    def test_health_check_response_model_validation(self, client):
        """Test that health check response follows the expected model structure"""
        response = client.get("/v1/health?check_dependencies=true")
        assert response.status_code == 200

        response_data = response.json()

        # Validate top-level structure
        required_fields = [
            "message",
            "databases",
            "object_storage",
            "nim",
            "processing",
            "task_management",
        ]
        for field in required_fields:
            assert field in response_data

        # Validate database health info structure
        for db_health in response_data["databases"]:
            required_db_fields = ["service", "url", "status", "latency_ms", "error"]
            for field in required_db_fields:
                assert field in db_health
            # Database-specific fields
            assert "collections" in db_health

        # Validate object storage health info structure
        for storage_health in response_data["object_storage"]:
            required_storage_fields = [
                "service",
                "url",
                "status",
                "latency_ms",
                "error",
            ]
            for field in required_storage_fields:
                assert field in storage_health
            # Storage-specific fields
            assert "buckets" in storage_health

        # Validate NIM service health info structure
        for nim_health in response_data["nim"]:
            required_nim_fields = [
                "service",
                "url",
                "status",
                "latency_ms",
                "error",
                "model",
            ]
            for field in required_nim_fields:
                assert field in nim_health

        # Validate processing service health info structure
        for processing_health in response_data["processing"]:
            required_processing_fields = [
                "service",
                "url",
                "status",
                "latency_ms",
                "error",
            ]
            for field in required_processing_fields:
                assert field in processing_health

        # Validate task management service health info structure
        for task_health in response_data["task_management"]:
            required_task_fields = ["service", "url", "status", "latency_ms", "error"]
            for field in required_task_fields:
                assert field in task_health

    def test_health_endpoint_tags_and_metadata(self, client):
        """Test that health endpoint is properly tagged and documented"""
        # This test verifies the endpoint exists and is accessible
        response = client.get("/v1/health")
        assert response.status_code == 200

        # Test with different parameter formats
        response = client.get("/v1/health", params={"check_dependencies": "true"})
        assert response.status_code == 200

        response = client.get("/v1/health", params={"check_dependencies": "false"})
        assert response.status_code == 200


class TestUploadDocumentsEndpoint:
    """Tests for the /documents POST endpoint"""

    def test_upload_documents_success(self, client):
        # Create a new file for this test
        sample_file = BytesIO(b"Test document content")
        files = {"documents": ("test.txt", sample_file, "text/plain")}
        data = {
            "collection_name": "test_collection",
            "blocking": False,
            "split_options": {"chunk_size": 512, "chunk_overlap": 150},
            "custom_metadata": [],
            "generate_summary": False,
        }

        response = client.post(
            "/v1/documents", files=files, data={"data": json.dumps(data)}
        )
        assert response.status_code == 200
        response_data = response.json()
        assert "message" in response_data
        assert (
            "task_id" in response_data
        )  # When blocking=False, returns task_id instead of document details

    def test_upload_documents_no_files(self, client):
        response = client.post("/v1/documents")
        assert response.status_code == 422  # Validation error for missing files

    def test_upload_documents_error(self, client):
        mock_nvidia_rag_ingestor_instance.raise_upload_error()
        # Create a new file for this test
        sample_file = BytesIO(b"Test document content")
        files = {"documents": ("test.txt", sample_file, "text/plain")}
        data = {
            "collection_name": "test_collection",
            "blocking": False,
            "split_options": {"chunk_size": 512, "chunk_overlap": 150},
            "custom_metadata": [],
            "generate_summary": False,
        }

        response = client.post(
            "/v1/documents", files=files, data={"data": json.dumps(data)}
        )
        assert response.status_code == 500
        assert "Upload failed" in response.json()["message"]

    def test_upload_documents_validation_error(self, client):
        mock_nvidia_rag_ingestor_instance.raise_upload_validation_error()
        # Create a new file for this test
        sample_file = BytesIO(b"Test document content")
        files = {"documents": ("test.txt", sample_file, "text/plain")}
        data = {
            "collection_name": "test_collection",
            "blocking": False,
            "split_options": {"chunk_size": 512, "chunk_overlap": 150},
            "custom_metadata": [],
            "generate_summary": False,
        }

        response = client.post(
            "/v1/documents", files=files, data={"data": json.dumps(data)}
        )
        assert response.status_code == 500  # Server returns 500 for ValueError
        assert "Invalid upload data" in response.json()["message"]


class TestGetStatusEndpoint:
    """Tests for the /status GET endpoint"""

    def test_get_status_success(self, client):
        response = client.get("/v1/status?task_id=test-task-id")
        assert response.status_code == 200
        response_data = response.json()
        assert "state" in response_data
        assert "result" in response_data
        assert response_data["state"] == "FINISHED"

    def test_get_status_no_task_id(self, client):
        response = client.get("/v1/status")
        assert response.status_code == 422

    def test_get_status_error(self, client):
        mock_nvidia_rag_ingestor_instance.raise_status_error()

        response = client.get("/v1/status?task_id=test-task-id")
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["state"] == "FAILED"
        assert response_data["result"]["message"] == "Status check failed"

    def test_get_status_not_found(self, client):
        mock_nvidia_rag_ingestor_instance.return_status_not_found()

        response = client.get("/v1/status?task_id=test-task-id")
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["state"] == "UNKNOWN"


class TestGetDocumentsEndpoint:
    """Tests for the /documents GET endpoint"""

    def test_get_documents_success(self, client):
        response = client.get("/v1/documents?collection_name=test_collection")
        assert response.status_code == 200
        response_data = response.json()
        assert "documents" in response_data
        assert "total_documents" in response_data
        assert "message" in response_data
        assert len(response_data["documents"]) > 0

    def test_get_documents_no_collection(self, client):
        response = client.get("/v1/documents")
        assert response.status_code == 200  # Server allows no collection parameter
        response_data = response.json()
        assert "documents" in response_data
        assert "total_documents" in response_data

    def test_get_documents_error(self, client):
        mock_nvidia_rag_ingestor_instance.raise_get_documents_error()

        response = client.get("/v1/documents?collection_name=test_collection")
        assert response.status_code == 500
        assert "Failed to get documents" in response.json()["message"]

    def test_get_documents_empty(self, client):
        mock_nvidia_rag_ingestor_instance.return_empty_documents()

        response = client.get("/v1/documents?collection_name=test_collection")
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["total_documents"] == 0


class TestGetCollectionsEndpoint:
    """Tests for the /collections GET endpoint"""

    def test_get_collections_success(self, client):
        response = client.get("/v1/collections")
        assert response.status_code == 200
        response_data = response.json()
        assert "collections" in response_data
        assert "total_collections" in response_data
        assert "message" in response_data
        assert len(response_data["collections"]) > 0

    def test_get_collections_error(self, client):
        mock_nvidia_rag_ingestor_instance.raise_get_collections_error()

        response = client.get("/v1/collections")
        assert response.status_code == 500
        assert "Failed to get collections" in response.json()["message"]

    def test_get_collections_empty(self, client):
        mock_nvidia_rag_ingestor_instance.return_empty_collections()

        response = client.get("/v1/collections")
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["total_collections"] == 0


class TestCreateCollectionEndpoint:
    """Tests for the /collections POST endpoint"""

    def test_create_collection_success(self, client):
        collection_data = {
            "collection_name": "new_collection",
            "vdb_endpoint": "http://localhost:19530",
            "embedding_dimension": 2048,
            "metadata_schema": [],
        }

        response = client.post("/v1/collection", json=collection_data)
        assert response.status_code == 200
        response_data = response.json()
        assert "message" in response_data
        assert "collection_name" in response_data

    def test_create_collection_invalid_data(self, client):
        collection_data = {
            "collection_name": "",  # Invalid empty name
            "vdb_endpoint": "http://localhost:19530",
            "embedding_dimension": 2048,
            "metadata_schema": [],
        }

        response = client.post("/v1/collection", json=collection_data)
        assert (
            response.status_code == 200
        )  # Server accepts empty collection name and handles it gracefully


class TestDeleteCollectionEndpoint:
    """Tests for the /collections DELETE endpoint"""

    def test_delete_collection_success(self, client):
        response = client.delete("/v1/collections?collection_names=test_collection")
        assert response.status_code == 200
        response_data = response.json()
        assert "message" in response_data
        assert "successful" in response_data
        assert "failed" in response_data

    def test_delete_collection_no_name(self, client):
        response = client.delete("/v1/collections")
        # Server returns 200 with empty lists when no collection names provided
        assert response.status_code == 200


class TestDeleteDocumentsEndpoint:
    """Tests for the /documents DELETE endpoint"""

    def test_delete_documents_success(self, client):
        response = client.delete(
            "/v1/documents?document_names=doc-1&document_names=doc-2&collection_name=test_collection"
        )
        assert response.status_code == 200
        response_data = response.json()
        assert "message" in response_data
        assert "total_documents" in response_data
        assert "documents" in response_data

    def test_delete_documents_invalid_data(self, client):
        response = client.delete("/v1/documents?collection_name=")  # Invalid empty name
        # Server returns 200 with empty results when collection name is empty
        assert response.status_code == 200


class TestPageFilterValidation:
    """Tests for page_filter validation in SummaryOptions Pydantic model"""

    def test_page_filter_simple_range(self):
        """Test simple page range validation"""
        so = SummaryOptions(page_filter=[[1, 10]])
        assert so.page_filter == [[1, 10]]

    def test_page_filter_multiple_ranges(self):
        """Test multiple page ranges"""
        so = SummaryOptions(page_filter=[[1, 10], [20, 30]])
        assert so.page_filter == [[1, 10], [20, 30]]

    def test_page_filter_negative_range(self):
        """Test negative (Pythonic) page range"""
        so = SummaryOptions(page_filter=[[-10, -1]])
        assert so.page_filter == [[-10, -1]]

    def test_page_filter_mixed_ranges(self):
        """Test mixing positive and negative ranges (different ranges, not same range)"""
        so = SummaryOptions(page_filter=[[1, 10], [-5, -1]])
        assert so.page_filter == [[1, 10], [-5, -1]]

    def test_page_filter_even_string(self):
        """Test 'even' string filter"""
        so = SummaryOptions(page_filter="even")
        assert so.page_filter == "even"  # normalized to lowercase

    def test_page_filter_odd_string(self):
        """Test 'odd' string filter"""
        so = SummaryOptions(page_filter="odd")
        assert so.page_filter == "odd"

    def test_page_filter_case_insensitive_string(self):
        """Test case-insensitive string normalization"""
        so = SummaryOptions(page_filter="EVEN")
        assert so.page_filter == "even"

        so = SummaryOptions(page_filter="ODD")
        assert so.page_filter == "odd"

    def test_page_filter_invalid_string(self):
        """Test invalid string value"""
        with pytest.raises(ValidationError, match="Invalid page_filter string"):
            SummaryOptions(page_filter="invalid")

    def test_page_filter_zero_page_rejected(self):
        """Test that page number 0 is rejected"""
        with pytest.raises(ValidationError, match="page numbers cannot be 0"):
            SummaryOptions(page_filter=[[0, 10]])

        with pytest.raises(ValidationError, match="page numbers cannot be 0"):
            SummaryOptions(page_filter=[[1, 0]])

    def test_page_filter_reversed_positive_range_rejected(self):
        """Test that reversed positive range is rejected"""
        with pytest.raises(ValidationError, match="start must be <= end"):
            SummaryOptions(page_filter=[[10, 1]])

    def test_page_filter_reversed_negative_range_rejected(self):
        """Test that reversed negative range is rejected"""
        with pytest.raises(ValidationError, match="invalid negative range"):
            SummaryOptions(page_filter=[[-1, -10]])

    def test_page_filter_mixed_positive_negative_same_range_rejected(self):
        """Test that mixing positive and negative in same range is rejected"""
        with pytest.raises(ValidationError, match="cannot mix positive and negative"):
            SummaryOptions(page_filter=[[-10, 1]])

        with pytest.raises(ValidationError, match="cannot mix positive and negative"):
            SummaryOptions(page_filter=[[1, -10]])

    def test_page_filter_empty_list_rejected(self):
        """Test that empty list is rejected"""
        with pytest.raises(
            ValidationError, match="Page filter range list cannot be empty"
        ):
            SummaryOptions(page_filter=[])

    def test_page_filter_non_list_range_rejected(self):
        """Test that non-list items are rejected"""
        with pytest.raises(
            ValidationError,
            match="Input should be a valid list|Input should be a valid string",
        ):
            SummaryOptions(page_filter=[1, 2, 3])

    def test_page_filter_wrong_range_size_rejected(self):
        """Test that ranges without exactly 2 elements are rejected"""
        with pytest.raises(ValidationError, match="must have exactly 2 elements"):
            SummaryOptions(page_filter=[[1]])

        with pytest.raises(ValidationError, match="must have exactly 2 elements"):
            SummaryOptions(page_filter=[[1, 2, 3]])


class TestSummaryOptionsValidation:
    """Tests for SummaryOptions Pydantic model"""

    def test_summary_options_with_page_filter(self):
        """Test SummaryOptions with page filter"""
        so = SummaryOptions(page_filter=[[1, 10]])
        assert so.page_filter == [[1, 10]]

    def test_summary_options_without_page_filter(self):
        """Test SummaryOptions without page filter"""
        so = SummaryOptions()
        assert so.page_filter is None

    def test_summary_options_none_page_filter(self):
        """Test SummaryOptions with explicit None"""
        so = SummaryOptions(page_filter=None)
        assert so.page_filter is None

    def test_summary_options_shallow_summary_field(self):
        """Test shallow_summary field with different values and combinations"""
        # Test default value (False)
        so = SummaryOptions()
        assert so.shallow_summary is False

        # Test explicit True
        so = SummaryOptions(shallow_summary=True)
        assert so.shallow_summary is True

        # Test explicit False
        so = SummaryOptions(shallow_summary=False)
        assert so.shallow_summary is False

        # Test with page filter combination
        so = SummaryOptions(page_filter=[[1, 10]], shallow_summary=True)
        assert so.page_filter == [[1, 10]]
        assert so.shallow_summary is True


class TestDocumentUploadRequestValidation:
    """Tests for DocumentUploadRequest cross-field validation"""

    def test_document_upload_no_summary(self):
        """Test request without summary generation"""
        req = DocumentUploadRequest(collection_name="test", generate_summary=False)
        assert req.generate_summary is False
        assert req.summary_options is None

    def test_document_upload_summary_all_pages(self):
        """Test request with summary for all pages"""
        req = DocumentUploadRequest(collection_name="test", generate_summary=True)
        assert req.generate_summary is True
        assert req.summary_options is None

    def test_document_upload_summary_with_filter(self):
        """Test request with summary and page filter"""
        req = DocumentUploadRequest(
            collection_name="test",
            generate_summary=True,
            summary_options=SummaryOptions(page_filter=[[1, 10]]),
        )
        assert req.generate_summary is True
        assert req.summary_options.page_filter == [[1, 10]]

    def test_document_upload_summary_options_without_generate_summary_rejected(self):
        """Test that summary_options without generate_summary is rejected"""
        with pytest.raises(
            ValidationError,
            match="summary_options can only be provided when generate_summary=True",
        ):
            DocumentUploadRequest(
                collection_name="test",
                generate_summary=False,
                summary_options=SummaryOptions(page_filter=[[1, 10]]),
            )

    def test_document_upload_empty_summary_options_without_generate_summary_rejected(
        self,
    ):
        """Test that even empty summary_options without generate_summary is rejected"""
        with pytest.raises(
            ValidationError,
            match="summary_options can only be provided when generate_summary=True",
        ):
            DocumentUploadRequest(
                collection_name="test",
                generate_summary=False,
                summary_options=SummaryOptions(),
            )


class TestSummaryOptionsStrategyValidation:
    """Tests for SummaryOptions validation, including summarization_strategy"""

    def test_summary_options_no_strategy(self):
        """Test SummaryOptions without strategy (uses default iterative)"""
        opts = SummaryOptions()
        assert opts.summarization_strategy is None
        assert opts.shallow_summary is False

    def test_summary_options_strategy_single(self):
        """Test SummaryOptions with 'single' strategy"""
        opts = SummaryOptions(summarization_strategy="single")
        assert opts.summarization_strategy == "single"

    def test_summary_options_strategy_hierarchical(self):
        """Test SummaryOptions with 'hierarchical' strategy"""
        opts = SummaryOptions(summarization_strategy="hierarchical")
        assert opts.summarization_strategy == "hierarchical"

    def test_summary_options_invalid_strategy_rejected(self):
        """Test that invalid summarization_strategy is rejected"""
        with pytest.raises(
            ValidationError,
            match="Invalid summarization_strategy: 'invalid_strategy'. Allowed values: \\['single', 'hierarchical'\\]",
        ):
            SummaryOptions(summarization_strategy="invalid_strategy")

    def test_summary_options_strategy_case_sensitive(self):
        """Test that strategy validation is case-sensitive"""
        with pytest.raises(
            ValidationError,
            match="Invalid summarization_strategy",
        ):
            SummaryOptions(summarization_strategy="Single")  # Capital S should fail

    def test_summary_options_with_page_filter_and_strategy(self):
        """Test SummaryOptions with both page_filter and summarization_strategy"""
        opts = SummaryOptions(
            page_filter=[[1, 5]],
            summarization_strategy="hierarchical",
        )
        assert opts.page_filter == [[1, 5]]
        assert opts.summarization_strategy == "hierarchical"

    def test_summary_options_with_shallow_and_strategy(self):
        """Test SummaryOptions with both shallow_summary and summarization_strategy"""
        opts = SummaryOptions(
            shallow_summary=True,
            summarization_strategy="single",
        )
        assert opts.shallow_summary is True
        assert opts.summarization_strategy == "single"

    def test_summary_options_all_features_combined(self):
        """Test SummaryOptions with all features: page_filter, shallow_summary, and strategy"""
        opts = SummaryOptions(
            page_filter="odd",
            shallow_summary=True,
            summarization_strategy="hierarchical",
        )
        assert opts.page_filter == "odd"
        assert opts.shallow_summary is True
        assert opts.summarization_strategy == "hierarchical"

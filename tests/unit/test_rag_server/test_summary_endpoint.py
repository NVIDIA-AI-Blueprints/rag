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

"""
Unit tests for the summary endpoint in the RAG server.

This module tests the /v1/summary endpoint functionality including:
- Timeout parameter validation (negative values)
- Valid timeout values
- Different blocking modes
- Error handling scenarios
- Success responses
"""

import json
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from nvidia_rag.rag_server.response_generator import ErrorCodeMapping


class MockNvidiaRAGSummary:
    """Mock class for NvidiaRAG summary functionality with configurable responses"""

    def __init__(self):
        self.reset()

    def reset(self):
        self._get_summary_side_effect = None
        self._get_summary_return_value = None

    def set_get_summary_success(
        self,
        summary_text="Test summary",
        file_name="test.pdf",
        collection_name="test_collection",
    ):
        """Set up successful summary response"""
        self._get_summary_return_value = {
            "message": "Summary retrieved successfully.",
            "summary": summary_text,
            "file_name": file_name,
            "collection_name": collection_name,
            "status": "SUCCESS",
        }
        self._get_summary_side_effect = None

    def set_get_summary_failed(self, file_name="test.pdf"):
        """Set up failed summary response (not found)"""
        self._get_summary_return_value = {
            "message": f"Summary for {file_name} not found. To generate a summary, upload the document with generate_summary=true.",
            "status": "NOT_FOUND",
            "file_name": file_name,
            "collection_name": "test_collection",
        }
        self._get_summary_side_effect = None

    def set_get_summary_pending(self, file_name="test.pdf"):
        """Set up pending summary response"""
        self._get_summary_return_value = {
            "message": "Summary generation is pending. Set blocking=true to wait for completion.",
            "status": "PENDING",
            "file_name": file_name,
            "collection_name": "test_collection",
            "queued_at": "2025-01-24T10:30:00.000Z",
        }
        self._get_summary_side_effect = None

    def set_get_summary_in_progress(self, file_name="test.pdf", current=3, total=5):
        """Set up in-progress summary response with chunk progress"""
        self._get_summary_return_value = {
            "message": "Summary generation is in progress. Set blocking=true to wait for completion.",
            "status": "IN_PROGRESS",
            "file_name": file_name,
            "collection_name": "test_collection",
            "started_at": "2025-01-24T10:30:00.000Z",
            "updated_at": "2025-01-24T10:30:15.000Z",
            "progress": {
                "current": current,
                "total": total,
                "message": f"Processing chunk {current}/{total}",
            },
        }
        self._get_summary_side_effect = None

    def set_get_summary_timeout(self, file_name="test.pdf"):
        """Set up timeout summary response (now returns FAILED with timeout error)"""
        self._get_summary_return_value = {
            "message": f"Timeout waiting for summary generation for {file_name} after 300 seconds",
            "status": "FAILED",
            "error": "Timeout after 300 seconds",
            "file_name": file_name,
            "collection_name": "test_collection",
        }
        self._get_summary_side_effect = None

    def set_get_summary_error(self, error_message="Internal server error"):
        """Set up error summary response (now returns FAILED status)"""
        self._get_summary_return_value = {
            "message": "Error occurred while getting summary.",
            "error": error_message,
            "status": "FAILED",
        }
        self._get_summary_side_effect = None

    def set_get_summary_exception(self, exception):
        """Set up exception to be raised during summary retrieval"""
        self._get_summary_side_effect = exception
        self._get_summary_return_value = None

    async def get_summary(
        self,
        collection_name: str,
        file_name: str,
        blocking: bool = False,
        timeout: int = 300,
    ):
        """Mock get_summary method"""
        if self._get_summary_side_effect:
            raise self._get_summary_side_effect

        return self._get_summary_return_value


# Global mock instance
mock_nvidia_rag_summary = MockNvidiaRAGSummary()


@pytest.fixture
def client():
    """Create test client with mocked dependencies"""
    with patch("nvidia_rag.rag_server.server.NVIDIA_RAG", mock_nvidia_rag_summary):
        from nvidia_rag.rag_server.server import app

        with TestClient(app) as test_client:
            yield test_client


@pytest.fixture
def valid_summary_params():
    """Valid parameters for summary endpoint"""
    return {
        "collection_name": "test_collection",
        "file_name": "test.pdf",
        "blocking": False,
        "timeout": 300,
    }


class TestSummaryEndpointTimeoutValidation:
    """Tests for timeout parameter validation in summary endpoint"""

    def test_negative_timeout_returns_400(self, client, valid_summary_params):
        """Test that negative timeout values return 400 Bad Request"""
        params = valid_summary_params.copy()
        params["timeout"] = -1

        response = client.get("/v1/summary", params=params)

        assert response.status_code == ErrorCodeMapping.BAD_REQUEST
        response_data = response.json()
        assert "message" in response_data
        assert "Invalid timeout value" in response_data["message"]
        assert "non-negative integer" in response_data["message"]
        assert "error" in response_data
        assert response_data["error"] == "Provided timeout value: -1"

    def test_zero_timeout_is_valid(self, client, valid_summary_params):
        """Test that zero timeout is valid (edge case)"""
        mock_nvidia_rag_summary.set_get_summary_success()

        params = valid_summary_params.copy()
        params["timeout"] = 0

        response = client.get("/v1/summary", params=params)

        assert response.status_code == ErrorCodeMapping.SUCCESS
        response_data = response.json()
        assert response_data["status"] == "SUCCESS"

    def test_default_timeout_is_valid(self, client, valid_summary_params):
        """Test that default timeout (300) is valid"""
        mock_nvidia_rag_summary.set_get_summary_success()

        # Remove timeout to test default value
        params = {k: v for k, v in valid_summary_params.items() if k != "timeout"}

        response = client.get("/v1/summary", params=params)

        assert response.status_code == ErrorCodeMapping.SUCCESS
        response_data = response.json()
        assert response_data["status"] == "SUCCESS"


class TestSummaryEndpointSuccessScenarios:
    """Tests for successful summary endpoint scenarios"""

    def test_summary_success_response(self, client, valid_summary_params):
        """Test successful summary retrieval"""
        mock_nvidia_rag_summary.set_get_summary_success(
            summary_text="This is a test summary",
            file_name="test.pdf",
            collection_name="test_collection",
        )

        response = client.get("/v1/summary", params=valid_summary_params)

        assert response.status_code == ErrorCodeMapping.SUCCESS
        response_data = response.json()
        assert response_data["status"] == "SUCCESS"
        assert response_data["summary"] == "This is a test summary"
        assert response_data["file_name"] == "test.pdf"
        assert response_data["collection_name"] == "test_collection"
        assert "Summary retrieved successfully" in response_data["message"]

    def test_summary_with_blocking_true(self, client, valid_summary_params):
        """Test summary retrieval with blocking=True"""
        mock_nvidia_rag_summary.set_get_summary_success()

        params = valid_summary_params.copy()
        params["blocking"] = True

        response = client.get("/v1/summary", params=params)

        assert response.status_code == ErrorCodeMapping.SUCCESS
        response_data = response.json()
        assert response_data["status"] == "SUCCESS"

    def test_summary_with_custom_timeout(self, client, valid_summary_params):
        """Test summary retrieval with custom timeout value"""
        mock_nvidia_rag_summary.set_get_summary_success()

        params = valid_summary_params.copy()
        params["timeout"] = 120

        response = client.get("/v1/summary", params=params)

        assert response.status_code == ErrorCodeMapping.SUCCESS
        response_data = response.json()
        assert response_data["status"] == "SUCCESS"


class TestSummaryEndpointErrorScenarios:
    """Tests for error scenarios in summary endpoint"""

    def test_summary_not_found_returns_404(self, client, valid_summary_params):
        """Test summary not found returns 404"""
        mock_nvidia_rag_summary.set_get_summary_failed("test.pdf")

        response = client.get("/v1/summary", params=valid_summary_params)

        assert response.status_code == ErrorCodeMapping.NOT_FOUND
        response_data = response.json()
        assert response_data["status"] == "NOT_FOUND"
        assert "not found" in response_data["message"]

    def test_summary_timeout_returns_timeout_as_failed(
        self, client, valid_summary_params
    ):
        """Test summary timeout returns FAILED status with timeout error"""
        mock_nvidia_rag_summary.set_get_summary_timeout("test.pdf")

        response = client.get("/v1/summary", params=valid_summary_params)

        # Timeout is now returned as FAILED with error containing "timeout"
        assert response.status_code == ErrorCodeMapping.REQUEST_TIMEOUT
        response_data = response.json()
        assert response_data["status"] == "FAILED"
        assert "timeout" in response_data.get("error", "").lower()

    def test_summary_error_returns_500(self, client, valid_summary_params):
        """Test summary error returns 500"""
        mock_nvidia_rag_summary.set_get_summary_error("Database connection failed")

        response = client.get("/v1/summary", params=valid_summary_params)

        assert response.status_code == ErrorCodeMapping.INTERNAL_SERVER_ERROR
        response_data = response.json()
        assert response_data["status"] == "FAILED"
        assert "Error occurred" in response_data["message"]

    def test_summary_exception_returns_500(self, client, valid_summary_params):
        """Test summary exception returns 500"""
        mock_nvidia_rag_summary.set_get_summary_exception(Exception("Unexpected error"))

        response = client.get("/v1/summary", params=valid_summary_params)

        assert response.status_code == ErrorCodeMapping.INTERNAL_SERVER_ERROR
        response_data = response.json()
        assert "Error occurred while getting summary" in response_data["message"]
        assert "Unexpected error" in response_data["error"]


class TestSummaryEndpointParameterValidation:
    """Tests for parameter validation in summary endpoint"""

    def test_missing_collection_name_returns_422(self, client):
        """Test missing collection_name returns 422"""
        params = {
            "file_name": "test.pdf",
            "blocking": False,
            "timeout": 300,
        }

        response = client.get("/v1/summary", params=params)

        assert response.status_code == ErrorCodeMapping.UNPROCESSABLE_ENTITY

    def test_missing_file_name_returns_422(self, client):
        """Test missing file_name returns 422"""
        params = {
            "collection_name": "test_collection",
            "blocking": False,
            "timeout": 300,
        }

        response = client.get("/v1/summary", params=params)

        assert response.status_code == ErrorCodeMapping.UNPROCESSABLE_ENTITY

    def test_invalid_blocking_type_returns_422(self, client):
        """Test invalid blocking type returns 422"""
        params = {
            "collection_name": "test_collection",
            "file_name": "test.pdf",
            "blocking": "invalid_boolean",
            "timeout": 300,
        }

        response = client.get("/v1/summary", params=params)

        assert response.status_code == ErrorCodeMapping.UNPROCESSABLE_ENTITY

    def test_invalid_timeout_type_returns_422(self, client):
        """Test invalid timeout type returns 422"""
        params = {
            "collection_name": "test_collection",
            "file_name": "test.pdf",
            "blocking": False,
            "timeout": "not_a_number",
        }

        response = client.get("/v1/summary", params=params)

        assert response.status_code == ErrorCodeMapping.UNPROCESSABLE_ENTITY


class TestSummaryEndpointEdgeCases:
    """Tests for edge cases in summary endpoint"""

    def test_very_large_timeout_value(self, client, valid_summary_params):
        """Test very large timeout value"""
        mock_nvidia_rag_summary.set_get_summary_success()

        params = valid_summary_params.copy()
        params["timeout"] = 999999

        response = client.get("/v1/summary", params=params)

        assert response.status_code == ErrorCodeMapping.SUCCESS
        response_data = response.json()
        assert response_data["status"] == "SUCCESS"

    def test_float_timeout_converted_to_int(self, client, valid_summary_params):
        """Test that float timeout values are converted to int"""
        mock_nvidia_rag_summary.set_get_summary_success()

        params = valid_summary_params.copy()
        params["timeout"] = 300.5

        response = client.get("/v1/summary", params=params)

        assert response.status_code == ErrorCodeMapping.SUCCESS
        response_data = response.json()
        assert response_data["status"] == "SUCCESS"


class TestSummaryEndpointNewStatusValues:
    """Tests for new status values (PENDING, IN_PROGRESS, NOT_FOUND)"""

    def test_summary_pending_returns_202(self, client, valid_summary_params):
        """Test PENDING status returns 202 Accepted"""
        mock_nvidia_rag_summary.set_get_summary_pending("test.pdf")

        response = client.get("/v1/summary", params=valid_summary_params)

        assert response.status_code == ErrorCodeMapping.ACCEPTED
        response_data = response.json()
        assert response_data["status"] == "PENDING"
        assert "pending" in response_data["message"].lower()
        assert response_data["file_name"] == "test.pdf"
        assert "queued_at" in response_data

    def test_summary_in_progress_returns_202(self, client, valid_summary_params):
        """Test IN_PROGRESS status returns 202 Accepted"""
        mock_nvidia_rag_summary.set_get_summary_in_progress(
            "test.pdf", current=3, total=5
        )

        response = client.get("/v1/summary", params=valid_summary_params)

        assert response.status_code == ErrorCodeMapping.ACCEPTED
        response_data = response.json()
        assert response_data["status"] == "IN_PROGRESS"
        assert "in progress" in response_data["message"].lower()
        assert response_data["file_name"] == "test.pdf"
        assert "started_at" in response_data
        assert "updated_at" in response_data
        assert "progress" in response_data

    def test_summary_in_progress_includes_chunk_progress(
        self, client, valid_summary_params
    ):
        """Test IN_PROGRESS includes chunk-level progress information"""
        mock_nvidia_rag_summary.set_get_summary_in_progress(
            "test.pdf", current=2, total=7
        )

        response = client.get("/v1/summary", params=valid_summary_params)

        assert response.status_code == ErrorCodeMapping.ACCEPTED
        response_data = response.json()
        assert response_data["status"] == "IN_PROGRESS"
        assert "progress" in response_data
        assert response_data["progress"]["current"] == 2
        assert response_data["progress"]["total"] == 7
        assert "Processing chunk 2/7" in response_data["progress"]["message"]

    def test_summary_in_progress_first_chunk(self, client, valid_summary_params):
        """Test IN_PROGRESS for first chunk"""
        mock_nvidia_rag_summary.set_get_summary_in_progress(
            "test.pdf", current=1, total=5
        )

        response = client.get("/v1/summary", params=valid_summary_params)

        assert response.status_code == ErrorCodeMapping.ACCEPTED
        response_data = response.json()
        assert response_data["progress"]["current"] == 1
        assert response_data["progress"]["total"] == 5

    def test_summary_in_progress_last_chunk(self, client, valid_summary_params):
        """Test IN_PROGRESS for last chunk"""
        mock_nvidia_rag_summary.set_get_summary_in_progress(
            "test.pdf", current=5, total=5
        )

        response = client.get("/v1/summary", params=valid_summary_params)

        assert response.status_code == ErrorCodeMapping.ACCEPTED
        response_data = response.json()
        assert response_data["progress"]["current"] == 5
        assert response_data["progress"]["total"] == 5

    def test_summary_not_found_with_helper_message(self, client, valid_summary_params):
        """Test NOT_FOUND includes helpful message about generating summary"""
        mock_nvidia_rag_summary.set_get_summary_failed("test.pdf")

        response = client.get("/v1/summary", params=valid_summary_params)

        assert response.status_code == ErrorCodeMapping.NOT_FOUND
        response_data = response.json()
        assert response_data["status"] == "NOT_FOUND"
        assert "generate_summary=true" in response_data["message"]


class TestSummaryEndpointStatusTransitions:
    """Tests for different status value transitions and scenarios"""

    def test_pending_to_in_progress_transition(self, client, valid_summary_params):
        """Test querying during PENDING -> IN_PROGRESS transition"""
        # First query shows PENDING
        mock_nvidia_rag_summary.set_get_summary_pending("test.pdf")
        response1 = client.get("/v1/summary", params=valid_summary_params)
        assert response1.json()["status"] == "PENDING"

        # Second query shows IN_PROGRESS
        mock_nvidia_rag_summary.set_get_summary_in_progress(
            "test.pdf", current=1, total=5
        )
        response2 = client.get("/v1/summary", params=valid_summary_params)
        assert response2.json()["status"] == "IN_PROGRESS"

    def test_in_progress_chunk_progression(self, client, valid_summary_params):
        """Test chunk progress advancing from 1/5 to 5/5"""
        for chunk in range(1, 6):
            mock_nvidia_rag_summary.set_get_summary_in_progress(
                "test.pdf", current=chunk, total=5
            )
            response = client.get("/v1/summary", params=valid_summary_params)
            response_data = response.json()

            assert response_data["status"] == "IN_PROGRESS"
            assert response_data["progress"]["current"] == chunk
            assert response_data["progress"]["total"] == 5

    def test_in_progress_to_success_transition(self, client, valid_summary_params):
        """Test transition from IN_PROGRESS to SUCCESS"""
        # First query shows IN_PROGRESS
        mock_nvidia_rag_summary.set_get_summary_in_progress(
            "test.pdf", current=4, total=5
        )
        response1 = client.get("/v1/summary", params=valid_summary_params)
        assert response1.json()["status"] == "IN_PROGRESS"

        # Second query shows SUCCESS
        mock_nvidia_rag_summary.set_get_summary_success(summary_text="Final summary")
        response2 = client.get("/v1/summary", params=valid_summary_params)
        response_data = response2.json()
        assert response_data["status"] == "SUCCESS"
        assert "summary" in response_data

    def test_pending_to_failed_transition(self, client, valid_summary_params):
        """Test transition from PENDING to FAILED"""
        # First query shows PENDING
        mock_nvidia_rag_summary.set_get_summary_pending("test.pdf")
        response1 = client.get("/v1/summary", params=valid_summary_params)
        assert response1.json()["status"] == "PENDING"

        # Second query shows FAILED
        mock_nvidia_rag_summary.set_get_summary_error("LLM connection failed")
        response2 = client.get("/v1/summary", params=valid_summary_params)
        response_data = response2.json()
        assert response_data["status"] == "FAILED"
        assert "error" in response_data


class TestSummaryEndpointBlockingBehavior:
    """Tests for blocking parameter with new status values"""

    def test_blocking_with_pending_status(self, client, valid_summary_params):
        """Test blocking=true with PENDING status"""
        mock_nvidia_rag_summary.set_get_summary_pending("test.pdf")

        params = valid_summary_params.copy()
        params["blocking"] = True

        response = client.get("/v1/summary", params=params)

        # In blocking mode, should still return PENDING if not yet started
        assert response.status_code == ErrorCodeMapping.ACCEPTED
        response_data = response.json()
        assert response_data["status"] == "PENDING"

    def test_blocking_with_in_progress_status(self, client, valid_summary_params):
        """Test blocking=true with IN_PROGRESS status"""
        mock_nvidia_rag_summary.set_get_summary_in_progress(
            "test.pdf", current=2, total=5
        )

        params = valid_summary_params.copy()
        params["blocking"] = True

        response = client.get("/v1/summary", params=params)

        # In blocking mode with IN_PROGRESS, mock returns current progress
        assert response.status_code == ErrorCodeMapping.ACCEPTED
        response_data = response.json()
        assert response_data["status"] == "IN_PROGRESS"
        assert "progress" in response_data

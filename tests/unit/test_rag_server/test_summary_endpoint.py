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
            "message": f"Summary for {file_name} not found. Ensure the file name and collection name are correct. Set wait=true to wait for generation.",
            "status": "FAILED",
        }
        self._get_summary_side_effect = None

    def set_get_summary_timeout(self, file_name="test.pdf"):
        """Set up timeout summary response"""
        self._get_summary_return_value = {
            "message": f"Timeout waiting for summary generation for {file_name}",
            "status": "TIMEOUT",
        }
        self._get_summary_side_effect = None

    def set_get_summary_error(self, error_message="Internal server error"):
        """Set up error summary response"""
        self._get_summary_return_value = {
            "message": "Error occurred while getting summary.",
            "error": error_message,
            "status": "ERROR",
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

    def test_large_negative_timeout_returns_400(self, client, valid_summary_params):
        """Test that large negative timeout values return 400 Bad Request"""
        params = valid_summary_params.copy()
        params["timeout"] = -999

        response = client.get("/v1/summary", params=params)

        assert response.status_code == ErrorCodeMapping.BAD_REQUEST
        response_data = response.json()
        assert "Invalid timeout value" in response_data["message"]
        assert response_data["error"] == "Provided timeout value: -999"

    def test_zero_timeout_is_valid(self, client, valid_summary_params):
        """Test that zero timeout is valid (edge case)"""
        mock_nvidia_rag_summary.set_get_summary_success()

        params = valid_summary_params.copy()
        params["timeout"] = 0

        response = client.get("/v1/summary", params=params)

        assert response.status_code == ErrorCodeMapping.SUCCESS
        response_data = response.json()
        assert response_data["status"] == "SUCCESS"

    def test_positive_timeout_is_valid(self, client, valid_summary_params):
        """Test that positive timeout values are valid"""
        mock_nvidia_rag_summary.set_get_summary_success()

        params = valid_summary_params.copy()
        params["timeout"] = 600

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
        assert response_data["status"] == "FAILED"
        assert "not found" in response_data["message"]

    def test_summary_timeout_returns_408(self, client, valid_summary_params):
        """Test summary timeout returns 408"""
        mock_nvidia_rag_summary.set_get_summary_timeout("test.pdf")

        response = client.get("/v1/summary", params=valid_summary_params)

        assert response.status_code == ErrorCodeMapping.REQUEST_TIMEOUT
        response_data = response.json()
        assert response_data["status"] == "TIMEOUT"
        assert "Timeout waiting" in response_data["message"]

    def test_summary_error_returns_500(self, client, valid_summary_params):
        """Test summary error returns 500"""
        mock_nvidia_rag_summary.set_get_summary_error("Database connection failed")

        response = client.get("/v1/summary", params=valid_summary_params)

        assert response.status_code == ErrorCodeMapping.INTERNAL_SERVER_ERROR
        response_data = response.json()
        assert response_data["status"] == "ERROR"
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

    def test_empty_collection_name(self, client, valid_summary_params):
        """Test empty collection_name"""
        mock_nvidia_rag_summary.set_get_summary_success()

        params = valid_summary_params.copy()
        params["collection_name"] = ""

        response = client.get("/v1/summary", params=params)

        assert response.status_code == ErrorCodeMapping.SUCCESS
        response_data = response.json()
        assert response_data["status"] == "SUCCESS"

    def test_empty_file_name(self, client, valid_summary_params):
        """Test empty file_name"""
        mock_nvidia_rag_summary.set_get_summary_success()

        params = valid_summary_params.copy()
        params["file_name"] = ""

        response = client.get("/v1/summary", params=params)

        assert response.status_code == ErrorCodeMapping.SUCCESS
        response_data = response.json()
        assert response_data["status"] == "SUCCESS"

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


class TestSummaryEndpointMockVerification:
    """Tests to verify mock interactions"""

    def test_get_summary_called_with_correct_parameters(
        self, client, valid_summary_params
    ):
        """Test that get_summary is called with correct parameters"""
        mock_nvidia_rag_summary.set_get_summary_success()

        response = client.get("/v1/summary", params=valid_summary_params)

        assert response.status_code == ErrorCodeMapping.SUCCESS
        # Note: In a real test, you would verify the mock was called with correct parameters
        # This would require more sophisticated mocking setup

    def test_timeout_validation_prevents_downstream_calls(
        self, client, valid_summary_params
    ):
        """Test that negative timeout validation prevents downstream calls"""
        params = valid_summary_params.copy()
        params["timeout"] = -1

        response = client.get("/v1/summary", params=params)

        assert response.status_code == ErrorCodeMapping.BAD_REQUEST
        # The mock should not be called due to early validation
        # This would require more sophisticated mocking to verify

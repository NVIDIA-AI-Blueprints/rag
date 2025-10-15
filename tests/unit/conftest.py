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
Pytest configuration for NVIDIA RAG tests
"""

import atexit
import logging
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# OpenTelemetry imports (optional - may not be available in all environments)
try:
    from opentelemetry import metrics, trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False

# --------------------------------------------------------------------------------------------------
# Prepare import for minio operator to avoid actual Minio connection
# during testing
fake_minio_operator = types.ModuleType("minio_operator")


def mock_minio_operator_methods():
    return MagicMock()


fake_minio_operator.get_minio_operator = mock_minio_operator_methods
fake_minio_operator.get_unique_thumbnail_id_collection_prefix = (
    mock_minio_operator_methods
)
fake_minio_operator.get_unique_thumbnail_id_file_name_prefix = (
    mock_minio_operator_methods
)
fake_minio_operator.get_unique_thumbnail_id = mock_minio_operator_methods
fake_minio_operator.MinioOperator = MagicMock()
# Temporarily inject the fake module into sys.modules
sys.modules["nvidia_rag.utils.minio_operator"] = fake_minio_operator


# --------------------------------------------------------------------------------------------------
# OpenTelemetry cleanup to prevent logging errors during test teardown
def cleanup_opentelemetry():
    """Clean up OpenTelemetry resources to prevent logging errors during test teardown"""
    if not OPENTELEMETRY_AVAILABLE:
        return

    try:
        # Clean up metrics provider - just shutdown, don't override
        try:
            current_meter_provider = metrics.get_meter_provider()
            if hasattr(current_meter_provider, "shutdown"):
                current_meter_provider.shutdown()
        except Exception:
            pass

        # Get the current tracer provider and clean up processors
        current_provider = trace.get_tracer_provider()
        if isinstance(current_provider, TracerProvider):
            # Force shutdown of all span processors
            if hasattr(current_provider, "_active_span_processor"):
                processor = current_provider._active_span_processor
                if hasattr(processor, "shutdown"):
                    try:
                        processor.shutdown()
                    except Exception:
                        pass  # Ignore errors during cleanup

    except Exception:
        # Ignore any errors during cleanup
        pass


# Register cleanup function to run at exit
atexit.register(cleanup_opentelemetry)


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment and ensure proper cleanup"""
    # Suppress OpenTelemetry logging errors during tests
    logging.getLogger("opentelemetry.sdk.trace.export").setLevel(logging.CRITICAL)
    logging.getLogger("opentelemetry.metrics._internal").setLevel(logging.CRITICAL)
    logging.getLogger("opentelemetry.trace").setLevel(logging.CRITICAL)

    yield

    # Clean up after each test
    cleanup_opentelemetry()

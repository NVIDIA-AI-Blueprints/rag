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
"""This defines the health check for the services used by the ingestor-server.
1. check_service_health(): Check the health of a service endpoint asynchronously.
2. check_minio_health(): Check the health of the MinIO server.
3. check_nv_ingest_health(): Check the health of the NV-Ingest service.
4. check_redis_health(): Check the health of the Redis server.
5. check_all_services_health(): Check the health of all services used by the ingestor.
6. print_health_report(): Print the health report for the services used by the ingestor.
9. check_and_print_services_health(): Check the health of all services and print a report.
"""

import asyncio
import logging
import os
import time
from typing import Any, Optional
from urllib.parse import urlparse

import aiohttp
from elasticsearch import Elasticsearch
from pymilvus import connections, utility

from nvidia_rag.utils.common import get_config
from nvidia_rag.utils.minio_operator import MinioOperator
from nvidia_rag.utils.vdb.vdb_base import VDBRag

logger = logging.getLogger(__name__)


async def check_service_health(
    url: str,
    service_name: str,
    method: str = "GET",
    timeout: int = 5,
    headers: dict[str, str] | None = None,
    json_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Check health of a service endpoint asynchronously.

    Args:
        url: The endpoint URL to check
        service_name: Name of the service for reporting
        method: HTTP method to use (GET, POST, etc.)
        timeout: Request timeout in seconds
        headers: Optional HTTP headers
        json_data: Optional JSON payload for POST requests

    Returns:
        Dictionary with status information
    """
    start_time = time.time()
    status = {
        "service": service_name,
        "url": url,
        "status": "unknown",
        "latency_ms": 0,
        "error": None,
    }

    if not url:
        status["status"] = "skipped"
        status["error"] = "No URL provided"
        return status

    try:
        # Add scheme if missing
        if not url.startswith(("http://", "https://")):
            url = "http://" + url

        # Disable SSL verification for external endpoints
        # This only affects health monitoring UI, not actual LLM API calls
        connector = aiohttp.TCPConnector(ssl=False)

        async with aiohttp.ClientSession(connector=connector) as session:
            request_kwargs = {
                "timeout": aiohttp.ClientTimeout(total=timeout),
                "headers": headers or {},
            }

            if method.upper() == "POST" and json_data:
                request_kwargs["json"] = json_data

            async with getattr(session, method.lower())(
                url, **request_kwargs
            ) as response:
                status["status"] = "healthy" if response.status < 400 else "unhealthy"
                status["http_status"] = response.status
                status["latency_ms"] = round((time.time() - start_time) * 1000, 2)

    except TimeoutError:
        status["status"] = "timeout"
        status["error"] = f"Request timed out after {timeout}s"
    except aiohttp.ClientError as e:
        status["status"] = "error"
        status["error"] = str(e)
    except Exception as e:
        status["status"] = "error"
        status["error"] = str(e)

    return status


async def check_minio_health(
    endpoint: str, access_key: str, secret_key: str
) -> dict[str, Any]:
    """Check MinIO server health"""
    status = {"service": "MinIO", "url": endpoint, "status": "unknown", "error": None}

    if not endpoint:
        status["status"] = "skipped"
        status["error"] = "No endpoint provided"
        return status

    try:
        start_time = time.time()
        minio_operator = MinioOperator(
            endpoint=endpoint, access_key=access_key, secret_key=secret_key
        )
        # Test basic operation - list buckets
        buckets = minio_operator.client.list_buckets()
        status["status"] = "healthy"
        status["latency_ms"] = round((time.time() - start_time) * 1000, 2)
        status["buckets"] = len(buckets)
    except Exception as e:
        status["status"] = "error"
        status["error"] = str(e)

    return status


async def check_nv_ingest_health(hostname: str, port: int) -> dict[str, Any]:
    """Check NV-Ingest service health"""
    status = {
        "service": "NV-Ingest",
        "url": f"{hostname}:{port}",
        "status": "unknown",
        "error": None,
    }

    if not hostname or not port:
        status["status"] = "skipped"
        status["error"] = "No hostname or port provided"
        return status

    try:
        start_time = time.time()

        # Check if NV-Ingest service is accessible
        # NV-Ingest typically exposes a health endpoint or we can check basic connectivity
        url = f"http://{hostname}:{port}/v1/health/ready"

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status < 400:
                        status["status"] = "healthy"
                        status["http_status"] = response.status
                    else:
                        status["status"] = "unhealthy"
                        status["http_status"] = response.status
            except aiohttp.ClientError:
                # If health endpoint doesn't exist, try basic connectivity check
                url = f"http://{hostname}:{port}"
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    # Any response indicates service is running
                    status["status"] = "healthy"
                    status["http_status"] = response.status

        status["latency_ms"] = round((time.time() - start_time) * 1000, 2)

    except Exception as e:
        status["status"] = "error"
        status["error"] = str(e)

    return status


async def check_redis_health(host: str, port: int, db: int) -> dict[str, Any]:
    """Check Redis server health"""
    status = {
        "service": "Redis",
        "url": f"{host}:{port}",
        "status": "unknown",
        "error": None,
    }

    if not host or not port:
        status["status"] = "skipped"
        status["error"] = "No host or port provided"
        return status

    try:
        start_time = time.time()

        # Try to import Redis
        from redis import Redis

        # Create Redis client
        redis_client = Redis(host=host, port=port, db=db)

        # Test basic operation - ping
        result = redis_client.ping()

        if result:
            status["status"] = "healthy"
            status["latency_ms"] = round((time.time() - start_time) * 1000, 2)
        else:
            status["status"] = "unhealthy"
            status["error"] = "Redis ping failed"

    except ImportError:
        status["status"] = "skipped"
        status["error"] = "Redis not available (library not installed)"
    except Exception as e:
        status["status"] = "error"
        status["error"] = str(e)

    return status


def is_nvidia_api_catalog_url(url: str) -> bool:
    """Check if the URL is from NVIDIA API Catalog"""
    if not url:
        return True
    return any(
        url.startswith(prefix)
        for prefix in [
            "https://integrate.api.nvidia.com",
            "https://ai.api.nvidia.com",
            "https://api.nvcf.nvidia.com",
        ]
    )


async def check_all_services_health(vdb_op: VDBRag) -> dict[str, list[dict[str, Any]]]:
    """
    Check health of all services used by the ingestor server

    Returns:
        Dictionary with service categories and their health status
    """
    config = get_config()

    # Create tasks for different service types
    tasks = []
    results = {
        "databases": [],
        "object_storage": [],
        "nim": [],  # NIM services (embeddings, LLM)
        "processing": [],  # Document processing services
        "task_management": [],  # Task management services
    }

    # MinIO health check
    minio_endpoint = config.minio.endpoint
    minio_access_key = config.minio.access_key
    minio_secret_key = config.minio.secret_key
    if minio_endpoint:
        tasks.append(
            (
                "object_storage",
                check_minio_health(
                    endpoint=minio_endpoint,
                    access_key=minio_access_key,
                    secret_key=minio_secret_key,
                ),
            )
        )

    # Vector DB health check
    try:
        tasks.append(("databases", vdb_op.check_health()))
    except Exception as e:
        logger.error(f"Error checking vector store health: {e}")
        # Unknown vector store type
        results["databases"].append(
            {
                "service": "Vector Store",
                "url": "Not configured",
                "status": "unknown",
                "error": f"Error checking vector store health: {e}",
            }
        )

    # NV-Ingest service health check
    if (
        config.nv_ingest.message_client_hostname
        and config.nv_ingest.message_client_port
    ):
        tasks.append(
            (
                "processing",
                check_nv_ingest_health(
                    hostname=config.nv_ingest.message_client_hostname,
                    port=config.nv_ingest.message_client_port,
                ),
            )
        )

    # Embedding service health check
    if config.embeddings.server_url and not is_nvidia_api_catalog_url(
        config.embeddings.server_url
    ):
        embed_url = config.embeddings.server_url
        if not embed_url.startswith(("http://", "https://")):
            embed_url = f"http://{embed_url}/v1/health/ready"
        else:
            embed_url = f"{embed_url}/v1/health/ready"

        # For local services, we need to create a custom result with model info
        async def check_embed_health():
            result = await check_service_health(
                url=embed_url, service_name="Embeddings"
            )
            result["model"] = config.embeddings.model_name
            return result

        tasks.append(("nim", check_embed_health()))
    else:
        # When URL is empty or from API catalog, assume the service is running via API catalog
        results["nim"].append(
            {
                "service": "Embeddings",
                "model": config.embeddings.model_name,
                "url": config.embeddings.server_url,
                "status": "healthy",
                "latency_ms": 0,
                "message": "Using NVIDIA API Catalog",
            }
        )

    # LLM service health check (for summary generation)
    if config.summarizer.server_url and not is_nvidia_api_catalog_url(
        config.summarizer.server_url
    ):
        llm_url = config.summarizer.server_url
        if not llm_url.startswith(("http://", "https://")):
            llm_url = f"http://{llm_url}/v1/health/ready"
        else:
            llm_url = f"{llm_url}/v1/health/ready"

        # For local services, we need to create a custom result with model info
        async def check_summary_llm_health():
            result = await check_service_health(url=llm_url, service_name="Summary LLM")
            result["model"] = config.summarizer.model_name
            return result

        tasks.append(("nim", check_summary_llm_health()))
    else:
        # When URL is empty or from API catalog, assume the service is running via API catalog
        results["nim"].append(
            {
                "service": "Summary LLM",
                "model": config.summarizer.model_name,
                "url": config.summarizer.server_url,
                "status": "healthy",
                "latency_ms": 0,
                "message": "Using NVIDIA API Catalog",
            }
        )

    # Caption model health check (only when image extraction is enabled)
    if config.nv_ingest.extract_images:
        if config.nv_ingest.caption_endpoint_url and not is_nvidia_api_catalog_url(
            config.nv_ingest.caption_endpoint_url
        ):
            caption_url = config.nv_ingest.caption_endpoint_url
            if not caption_url.startswith(("http://", "https://")):
                caption_url = f"http://{caption_url}/v1/health/ready"
            else:
                # For caption endpoints, try health endpoint first, fall back to base URL
                if caption_url.endswith("/v1/chat/completions"):
                    caption_url = caption_url.replace(
                        "/v1/chat/completions", "/v1/health/ready"
                    )
                elif not caption_url.endswith("/v1/health/ready"):
                    caption_url = f"{caption_url}/v1/health/ready"

            # For local services, we need to create a custom result with model info
            async def check_caption_health():
                result = await check_service_health(
                    url=caption_url, service_name="Caption Model"
                )
                result["model"] = config.nv_ingest.caption_model_name
                return result

            tasks.append(("nim", check_caption_health()))
        else:
            # When URL is empty or from API catalog, assume the service is running via API catalog
            results["nim"].append(
                {
                    "service": "Caption Model",
                    "model": config.nv_ingest.caption_model_name,
                    "url": config.nv_ingest.caption_endpoint_url
                    if config.nv_ingest.caption_endpoint_url
                    else "Not configured",
                    "status": "healthy",
                    "latency_ms": 0,
                    "message": "Using NVIDIA API Catalog"
                    if config.nv_ingest.caption_endpoint_url
                    else "Using NVIDIA API Catalog (default)",
                }
            )

    # Redis health check (for task management)
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", 6379))
    redis_db = int(os.getenv("REDIS_DB", 0))
    tasks.append(
        (
            "task_management",
            check_redis_health(host=redis_host, port=redis_port, db=redis_db),
        )
    )

    # Execute all health checks concurrently
    for category, task in tasks:
        result = await task
        results[category].append(result)

    return results


def print_health_report(health_results: dict[str, list[dict[str, Any]]]) -> None:
    """
    Print health status for individual services

    Args:
        health_results: Results from check_all_services_health
    """
    logger.info("===== INGESTOR SERVICE HEALTH STATUS =====")

    for category, services in health_results.items():
        if not services or isinstance(services, str):
            continue

        category_name = category.replace("_", " ").title()
        logger.info(f"--- {category_name} ---")

        for service in services:
            if service["status"] == "healthy":
                logger.info(
                    f"✓ {service['service']} is healthy - Response time: {service.get('latency_ms', 'N/A')}ms"
                )
            elif service["status"] == "skipped":
                logger.info(
                    f"- {service['service']} check skipped - Reason: {service.get('error', 'No URL provided')}"
                )
            else:
                error_msg = service.get("error", "Unknown error")
                logger.info(
                    f"✗ {service['service']} is not healthy - Issue: {error_msg}"
                )

    logger.info("=============================================")


async def check_and_print_services_health(vdb_op: VDBRag):
    """
    Check health of all services and print a report
    """
    health_results = await check_all_services_health(vdb_op)
    print_health_report(health_results)
    return health_results


def check_services_health(vdb_op: VDBRag):
    """
    Synchronous wrapper for checking service health
    """
    return asyncio.run(check_and_print_services_health(vdb_op))

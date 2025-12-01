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

from nvidia_rag.utils.configuration import NvidiaRAGConfig
from nvidia_rag.utils.health_models import (
    DatabaseHealthInfo,
    HealthResponseBase,
    IngestorHealthResponse,
    NIMServiceHealthInfo,
    ProcessingHealthInfo,
    ServiceStatus,
    StorageHealthInfo,
    TaskManagementHealthInfo,
)
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
) -> NIMServiceHealthInfo:
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
        NIMServiceHealthInfo with status information
    """
    start_time = time.time()
    
    if not url:
        return NIMServiceHealthInfo(
            service=service_name,
            url=url,
            status=ServiceStatus.SKIPPED,
            latency_ms=0,
            error="No URL provided"
        )

    http_status = None
    status = ServiceStatus.UNKNOWN
    error = None
    
    try:
        # Add scheme if missing
        if not url.startswith(("http://", "https://")):
            url = "http://" + url

        # SSL verification is ENABLED by default for security
        # NOTE: Only disable SSL verification in development/testing environments with self-signed certificates
        # To disable: Uncomment the line below and comment out the default connector
        # connector = aiohttp.TCPConnector(ssl=False)  # WARNING: Not secure for production!
        connector = aiohttp.TCPConnector(ssl=True)

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
                status = ServiceStatus.HEALTHY if response.status < 400 else ServiceStatus.UNHEALTHY
                http_status = response.status
                latency_ms = round((time.time() - start_time) * 1000, 2)
                
                return NIMServiceHealthInfo(
                    service=service_name,
                    url=url,
                    status=status,
                    latency_ms=latency_ms,
                    http_status=http_status
                )

    except TimeoutError:
        error = f"Request timed out after {timeout}s"
        return NIMServiceHealthInfo(
            service=service_name,
            url=url,
            status=ServiceStatus.TIMEOUT,
            latency_ms=0,
            error=error
        )
    except aiohttp.ClientError as e:
        return NIMServiceHealthInfo(
            service=service_name,
            url=url,
            status=ServiceStatus.ERROR,
            latency_ms=0,
            error=str(e)
        )
    except Exception as e:
        return NIMServiceHealthInfo(
            service=service_name,
            url=url,
            status=ServiceStatus.ERROR,
            latency_ms=0,
            error=str(e)
        )


async def check_minio_health(
    endpoint: str, access_key: str, secret_key: str
) -> StorageHealthInfo:
    """Check MinIO server health"""
    if not endpoint:
        return StorageHealthInfo(
            service="MinIO",
            url=endpoint,
            status=ServiceStatus.SKIPPED,
            latency_ms=0,
            error="No endpoint provided"
        )

    try:
        start_time = time.time()
        minio_operator = MinioOperator(
            endpoint=endpoint, access_key=access_key, secret_key=secret_key
        )
        # Test basic operation - list buckets
        buckets = minio_operator.client.list_buckets()
        latency_ms = round((time.time() - start_time) * 1000, 2)
        
        return StorageHealthInfo(
            service="MinIO",
            url=endpoint,
            status=ServiceStatus.HEALTHY,
            latency_ms=latency_ms,
            buckets=len(buckets)
        )
    except Exception as e:
        return StorageHealthInfo(
            service="MinIO",
            url=endpoint,
            status=ServiceStatus.ERROR,
            latency_ms=0,
            error=str(e)
        )


async def check_nv_ingest_health(hostname: str, port: int) -> ProcessingHealthInfo:
    """Check NV-Ingest service health"""
    url_base = f"{hostname}:{port}"
    
    if not hostname or not port:
        return ProcessingHealthInfo(
            service="NV-Ingest",
            url=url_base,
            status=ServiceStatus.SKIPPED,
            latency_ms=0,
            error="No hostname or port provided"
        )

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
                    latency_ms = round((time.time() - start_time) * 1000, 2)
                    if response.status < 400:
                        return ProcessingHealthInfo(
                            service="NV-Ingest",
                            url=url_base,
                            status=ServiceStatus.HEALTHY,
                            latency_ms=latency_ms,
                            http_status=response.status
                        )
                    else:
                        return ProcessingHealthInfo(
                            service="NV-Ingest",
                            url=url_base,
                            status=ServiceStatus.UNHEALTHY,
                            latency_ms=latency_ms,
                            http_status=response.status
                        )
            except aiohttp.ClientError:
                # If health endpoint doesn't exist, try basic connectivity check
                url = f"http://{hostname}:{port}"
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    latency_ms = round((time.time() - start_time) * 1000, 2)
                    # Any response indicates service is running
                    return ProcessingHealthInfo(
                        service="NV-Ingest",
                        url=url_base,
                        status=ServiceStatus.HEALTHY,
                        latency_ms=latency_ms,
                        http_status=response.status
                    )

    except Exception as e:
        return ProcessingHealthInfo(
            service="NV-Ingest",
            url=url_base,
            status=ServiceStatus.ERROR,
            latency_ms=0,
            error=str(e)
        )


async def check_redis_health(host: str, port: int, db: int) -> TaskManagementHealthInfo:
    """Check Redis server health"""
    url_base = f"{host}:{port}"
    
    if not host or not port:
        return TaskManagementHealthInfo(
            service="Redis",
            url=url_base,
            status=ServiceStatus.SKIPPED,
            latency_ms=0,
            error="No host or port provided"
        )

    try:
        start_time = time.time()

        # Try to import Redis
        from redis import Redis

        # Create Redis client
        redis_client = Redis(host=host, port=port, db=db)

        # Test basic operation - ping
        result = redis_client.ping()

        latency_ms = round((time.time() - start_time) * 1000, 2)
        
        if result:
            return TaskManagementHealthInfo(
                service="Redis",
                url=url_base,
                status=ServiceStatus.HEALTHY,
                latency_ms=latency_ms
            )
        else:
            return TaskManagementHealthInfo(
                service="Redis",
                url=url_base,
                status=ServiceStatus.UNHEALTHY,
                latency_ms=0,
                error="Redis ping failed"
            )

    except ImportError:
        return TaskManagementHealthInfo(
            service="Redis",
            url=url_base,
            status=ServiceStatus.SKIPPED,
            latency_ms=0,
            error="Redis not available (library not installed)"
        )
    except Exception as e:
        return TaskManagementHealthInfo(
            service="Redis",
            url=url_base,
            status=ServiceStatus.ERROR,
            latency_ms=0,
            error=str(e)
        )


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


async def check_all_services_health(
    vdb_op: VDBRag, config: NvidiaRAGConfig | None = None
) -> IngestorHealthResponse:
    """
    Check health of all services used by the ingestor server

    Args:
        vdb_op: Vector database operation instance
        config: NvidiaRAGConfig instance. If None, creates a new one.

    Returns:
        IngestorHealthResponse with service categories and their health status
    """
    if config is None:
        config = NvidiaRAGConfig()

    # Create tasks for different service types
    tasks = []
    databases: list[DatabaseHealthInfo] = []
    object_storage: list[StorageHealthInfo] = []
    nim: list[NIMServiceHealthInfo] = []
    processing: list[ProcessingHealthInfo] = []
    task_management: list[TaskManagementHealthInfo] = []

    # MinIO health check
    minio_endpoint = config.minio.endpoint
    minio_access_key = config.minio.access_key.get_secret_value()
    minio_secret_key = config.minio.secret_key.get_secret_value()
    if minio_endpoint:
        minio_result = await check_minio_health(
            endpoint=minio_endpoint,
            access_key=minio_access_key,
            secret_key=minio_secret_key,
        )
        object_storage.append(minio_result)

    # Vector DB health check
    try:
        tasks.append(("databases", vdb_op.check_health()))
    except Exception as e:
        logger.error(f"Error checking vector store health: {e}")
        # Unknown vector store type
        databases.append(
            DatabaseHealthInfo(
                service="Vector Store",
                url="Not configured",
                status=ServiceStatus.UNKNOWN,
                error=f"Error checking vector store health: {e}",
            )
        )

    # NV-Ingest service health check
    if (
        config.nv_ingest.message_client_hostname
        and config.nv_ingest.message_client_port
    ):
        nv_ingest_result = await check_nv_ingest_health(
            hostname=config.nv_ingest.message_client_hostname,
            port=config.nv_ingest.message_client_port,
        )
        processing.append(nv_ingest_result)

    # Embedding service health check
    if config.embeddings.server_url and not is_nvidia_api_catalog_url(
        config.embeddings.server_url
    ):
        embed_url = config.embeddings.server_url
        if not embed_url.startswith(("http://", "https://")):
            embed_url = f"http://{embed_url}/v1/health/ready"
        else:
            embed_url = f"{embed_url}/v1/health/ready"

        # For local services, check health and add model info
        embed_result = await check_service_health(url=embed_url, service_name="Embeddings")
        nim.append(embed_result.model_copy(update={"model": config.embeddings.model_name}))
    else:
        # When URL is empty or from API catalog, assume the service is running via API catalog
        nim.append(
            NIMServiceHealthInfo(
                service="Embeddings",
                model=config.embeddings.model_name,
                url=config.embeddings.server_url or "",
                status=ServiceStatus.HEALTHY,
                latency_ms=0,
                message="Using NVIDIA API Catalog",
            )
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

        # For local services, check health and add model info
        llm_result = await check_service_health(url=llm_url, service_name="Summary LLM")
        nim.append(llm_result.model_copy(update={"model": config.summarizer.model_name}))
    else:
        # When URL is empty or from API catalog, assume the service is running via API catalog
        nim.append(
            NIMServiceHealthInfo(
                service="Summary LLM",
                model=config.summarizer.model_name,
                url=config.summarizer.server_url or "",
                status=ServiceStatus.HEALTHY,
                latency_ms=0,
                message="Using NVIDIA API Catalog",
            )
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

            # For local services, check health and add model info
            caption_result = await check_service_health(url=caption_url, service_name="Caption Model")
            nim.append(caption_result.model_copy(update={"model": config.nv_ingest.caption_model_name}))
        else:
            # When URL is empty or from API catalog, assume the service is running via API catalog
            nim.append(
                NIMServiceHealthInfo(
                    service="Caption Model",
                    model=config.nv_ingest.caption_model_name,
                    url=config.nv_ingest.caption_endpoint_url or "Not configured",
                    status=ServiceStatus.HEALTHY,
                    latency_ms=0,
                    message="Using NVIDIA API Catalog"
                    if config.nv_ingest.caption_endpoint_url
                    else "Using NVIDIA API Catalog (default)",
                )
            )

    # Redis health check (for task management)
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", 6379))
    redis_db = int(os.getenv("REDIS_DB", 0))
    redis_result = await check_redis_health(host=redis_host, port=redis_port, db=redis_db)
    task_management.append(redis_result)

    # Execute all health checks concurrently for vector DB
    for category, task in tasks:
        result = await task
        if category == "databases":
            databases.append(DatabaseHealthInfo(**result))

    return IngestorHealthResponse(
        message="Service is up.",
        databases=databases,
        object_storage=object_storage,
        nim=nim,
        processing=processing,
        task_management=task_management,
    )


def print_health_report(health_results: IngestorHealthResponse) -> None:
    """
    Print health status for individual services

    Args:
        health_results: IngestorHealthResponse from check_all_services_health
    """
    logger.info("===== INGESTOR SERVICE HEALTH STATUS =====")

    # Combine all services into a single list for iteration
    # Use getattr with default empty list for fields that may not exist in all response types
    all_services = (
        health_results.databases
        + health_results.object_storage
        + health_results.nim
        + getattr(health_results, "processing", [])
        + getattr(health_results, "task_management", [])
    )

    for service in all_services:
        if service.status == ServiceStatus.HEALTHY or service.status == ServiceStatus.HEALTHY.value:
            logger.info(
                f"✓ {service.service} is healthy - Response time: {service.latency_ms}ms"
            )
        elif service.status == ServiceStatus.SKIPPED or service.status == ServiceStatus.SKIPPED.value:
            logger.info(
                f"- {service.service} check skipped - Reason: {service.error or 'No URL provided'}"
            )
        else:
            error_msg = service.error or "Unknown error"
            logger.info(
                f"✗ {service.service} is not healthy - Issue: {error_msg}"
            )

    logger.info("=============================================")


async def check_and_print_services_health(
    vdb_op: VDBRag, config: NvidiaRAGConfig | None = None
):
    """
    Check health of all services and print a report

    Args:
        vdb_op: Vector database operation instance
        config: NvidiaRAGConfig instance. If None, creates a new one.
    """
    health_results = await check_all_services_health(vdb_op, config)
    print_health_report(health_results)
    return health_results


def check_services_health(vdb_op: VDBRag):
    """
    Synchronous wrapper for checking service health
    """
    return asyncio.run(check_and_print_services_health(vdb_op))

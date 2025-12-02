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
"""This defines the health check for the services used by the rag-server.
1. check_service_health(): Check the health of a service endpoint asynchronously.
2. check_minio_health(): Check the health of the MinIO server.
3. check_all_services_health(): Check the health of all services used by the application.
4. print_health_report(): Print the health report for the services used by the application.
5. check_and_print_services_health(): Check the health of all services and print a report.
"""

import asyncio
import logging
import os
import time
from typing import Any
from urllib.parse import urlparse

import aiohttp
from elasticsearch import Elasticsearch
from pymilvus import connections, utility

from nvidia_rag.utils.configuration import NvidiaRAGConfig
from nvidia_rag.utils.health_models import (
    DatabaseHealthInfo,
    HealthResponseBase,
    NIMServiceHealthInfo,
    RAGHealthResponse,
    ServiceStatus,
    StorageHealthInfo,
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
) -> RAGHealthResponse:
    """
    Check health of all services used by the RAG application

    Args:
        vdb_op: Vector database operation instance
        config: NvidiaRAGConfig instance. If None, creates a new one.

    Returns:
        RAGHealthResponse with service categories and their health status
    """
    if config is None:
        config = NvidiaRAGConfig()

    # Create tasks for different service types
    tasks = []
    databases: list[DatabaseHealthInfo] = []
    object_storage: list[StorageHealthInfo] = []
    nim: list[NIMServiceHealthInfo] = []

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

    # LLM service health check
    if config.llm.server_url and not is_nvidia_api_catalog_url(config.llm.server_url):
        llm_url = config.llm.server_url
        if not llm_url.startswith(("http://", "https://")):
            llm_url = f"http://{llm_url}/v1/health/ready"
        else:
            llm_url = f"{llm_url}/v1/health/ready"

        # For local services, check health and add model info
        llm_result = await check_service_health(url=llm_url, service_name="LLM")
        nim.append(llm_result.model_copy(update={"model": config.llm.model_name}))
    else:
        # When URL is empty or from API catalog, assume the service is running
        # via API catalog
        nim.append(
            NIMServiceHealthInfo(
                service="LLM",
                model=config.llm.model_name,
                url=config.llm.server_url or "",
                status=ServiceStatus.HEALTHY,
                latency_ms=0,
                message="Using NVIDIA API Catalog",
            )
        )

    query_rewriter_enabled = config.query_rewriter.enable_query_rewriter

    if query_rewriter_enabled:
        # Query rewriter LLM health check
        if config.query_rewriter.server_url and not is_nvidia_api_catalog_url(
            config.query_rewriter.server_url
        ):
            qr_url = config.query_rewriter.server_url
            if not qr_url.startswith(("http://", "https://")):
                qr_url = f"http://{qr_url}/v1/health/ready"
            else:
                qr_url = f"{qr_url}/v1/health/ready"

            # For local services, check health and add model info
            qr_result = await check_service_health(url=qr_url, service_name="Query Rewriter")
            nim.append(qr_result.model_copy(update={"model": config.query_rewriter.model_name}))
        else:
            # When URL is empty or from API catalog, assume the service is
            # running via API catalog
            nim.append(
                NIMServiceHealthInfo(
                    service="Query Rewriter",
                    model=config.query_rewriter.model_name,
                    url=config.query_rewriter.server_url or "",
                    status=ServiceStatus.HEALTHY,
                    latency_ms=0,
                    message="Using NVIDIA API Catalog",
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

        # For local services, check health and add model info
        embed_result = await check_service_health(url=embed_url, service_name="Embeddings")
        nim.append(embed_result.model_copy(update={"model": config.embeddings.model_name}))
    else:
        # When URL is empty or from API catalog, assume the service is running
        # via API catalog
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

    enable_reranker = config.ranking.enable_reranker
    # Ranking service health check
    if enable_reranker:
        if config.ranking.server_url and not is_nvidia_api_catalog_url(
            config.ranking.server_url
        ):
            ranking_url = config.ranking.server_url
            if not ranking_url.startswith(("http://", "https://")):
                ranking_url = f"http://{ranking_url}/v1/health/ready"
            else:
                ranking_url = f"{ranking_url}/v1/health/ready"

            # For local services, check health and add model info
            ranking_result = await check_service_health(url=ranking_url, service_name="Ranking")
            nim.append(ranking_result.model_copy(update={"model": config.ranking.model_name}))
        else:
            # When URL is empty or from API catalog, assume the service is
            # running via API catalog
            nim.append(
                NIMServiceHealthInfo(
                    service="Ranking",
                    model=config.ranking.model_name,
                    url=config.ranking.server_url or "",
                    status=ServiceStatus.HEALTHY,
                    latency_ms=0,
                    message="Using NVIDIA API Catalog",
                )
            )

    # NemoGuardrails health check
    enable_guardrails = config.enable_guardrails
    if enable_guardrails:
        guardrails_url = os.getenv("NEMO_GUARDRAILS_URL", "")
        if guardrails_url:
            if not guardrails_url.startswith(("http://", "https://")):
                guardrails_url = f"http://{guardrails_url}/v1/health"
            else:
                guardrails_url = f"{guardrails_url}/v1/health"
            guardrails_result = await check_service_health(
                url=guardrails_url, service_name="NemoGuardrails"
            )
            nim.append(guardrails_result)
        else:
            nim.append(
                NIMServiceHealthInfo(
                    service="NemoGuardrails",
                    url="Not configured",
                    status=ServiceStatus.SKIPPED,
                    message="URL not provided",
                )
            )

    # Reflection LLM health check
    enable_reflection = os.getenv("ENABLE_REFLECTION", "False").lower() == "true"
    if enable_reflection:
        reflection_llm = os.getenv("REFLECTION_LLM", "").strip('"').strip("'")
        reflection_url = os.getenv("REFLECTION_LLM_SERVERURL", "").strip('"').strip("'")
        if reflection_url:
            if not reflection_url.startswith(("http://", "https://")):
                reflection_url = f"http://{reflection_url}/v1/health/ready"
            else:
                reflection_url = f"{reflection_url}/v1/health/ready"

            # For local services, check health and add model info
            reflection_result = await check_service_health(
                url=reflection_url, service_name="Reflection LLM"
            )
            nim.append(reflection_result.model_copy(update={"model": reflection_llm}))
        else:
            # When URL is empty, assume the service is running via API catalog
            nim.append(
                NIMServiceHealthInfo(
                    service="Reflection LLM",
                    model=reflection_llm,
                    url=os.getenv("REFLECTION_LLM_SERVERURL", "")
                    .strip('"')
                    .strip("'")
                    or "",
                    status=ServiceStatus.HEALTHY,
                    latency_ms=0,
                    message="Using NVIDIA API Catalog",
                )
            )

    # Execute all health checks concurrently for vector DB
    for category, task in tasks:
        result = await task
        if category == "databases":
            databases.append(DatabaseHealthInfo(**result))

    return RAGHealthResponse(
        message="Service is up.",
        databases=databases,
        object_storage=object_storage,
        nim=nim,
    )


def print_health_report(health_results: HealthResponseBase) -> None:
    """
    Print health status for individual services

    Args:
        health_results: HealthResponseBase (RAGHealthResponse or IngestorHealthResponse) from check_all_services_health
    """
    logger.info("===== SERVICE HEALTH STATUS =====")

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
                f"Service '{service.service}' is healthy - Response time: {service.latency_ms}ms"
            )
        elif service.status == ServiceStatus.SKIPPED or service.status == ServiceStatus.SKIPPED.value:
            logger.info(
                f"Service '{service.service}' check skipped - Reason: {service.error or 'No URL provided'}"
            )
        else:
            error_msg = service.error or "Unknown error"
            logger.info(
                f"Service '{service.service}' is not healthy - Issue: {error_msg}"
            )

    logger.info("================================")


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

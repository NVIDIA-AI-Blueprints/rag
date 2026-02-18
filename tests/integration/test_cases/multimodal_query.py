# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Multimodal Query integration tests

Tests for multimodal RAG queries that combine text and images.
Based on the notebook: notebooks/image_input.ipynb

IMPORTANT: This test requires a specific deployment configuration different from basic tests.
See docs/multimodal-query.md for full deployment instructions.

Required Environment Variables:
    # VLM Embedding Model (for multimodal embeddings)
    APP_EMBEDDINGS_MODELNAME=nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1
    APP_EMBEDDINGS_SERVERURL=https://integrate.api.nvidia.com/v1  # or on-prem URL

    # VLM Model (for multimodal generation)
    APP_VLM_MODELNAME=nvidia/nemotron-nano-12b-v2-vl
    APP_VLM_SERVERURL=https://integrate.api.nvidia.com/v1  # or on-prem URL
    ENABLE_VLM_INFERENCE=true
    VLM_TO_LLM_FALLBACK=false

    # Ingestion configuration (for image extraction)
    APP_NVINGEST_STRUCTURED_ELEMENTS_MODALITY=""
    APP_NVINGEST_IMAGE_ELEMENTS_MODALITY=image
    APP_NVINGEST_EXTRACTIMAGES=True

    # Disable reranker (not supported with multimodal queries)
    ENABLE_RERANKER=false
    APP_RANKING_SERVERURL=""

Docker Deployment:
    On-Prem (requires VLM NIMs):
        docker compose --profile vlm-ingest --profile vlm-only -f deploy/compose/nims.yaml up -d

    Cloud:
        Use NVIDIA-hosted endpoints as shown above.

Helm Deployment:
    On-prem deployment of VLM models requires an additional 1xH100 or 1xA100 GPU.

    1. Update deploy/helm/nvidia-blueprint-rag/values.yaml:

        # Enable VLM NIM for multimodal generation
        nim-vlm:
          enabled: true

        # Enable VLM embedding NIM for multimodal embeddings
        nvidia-nim-llama-32-nemoretriever-1b-vlm-embed-v1:
          enabled: true
          image:
            repository: nvcr.io/nvidia/nemo-microservices/llama-3.2-nemoretriever-1b-vlm-embed-v1
            tag: "1.7.0"

        # Optional: disable the default text embedding NIM
        nvidia-nim-llama-32-nv-embedqa-1b-v2:
          enabled: false

        # Disable LLM NIM (VLM handles generation)
        nim-llm:
          enabled: false

        # Configure environment variables
        envVars:
          # VLM inference settings
          ENABLE_VLM_INFERENCE: "true"
          VLM_TO_LLM_FALLBACK: "false"
          APP_VLM_MODELNAME: "nvidia/nemotron-nano-12b-v2-vl"
          APP_VLM_SERVERURL: "http://nim-vlm:8000/v1"

          # VLM embedding settings
          APP_EMBEDDINGS_SERVERURL: "nemoretriever-vlm-embedding-ms:8000"
          APP_EMBEDDINGS_MODELNAME: "nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1"

          # Disable reranker (not supported with multimodal queries)
          ENABLE_RERANKER: "False"
          APP_RANKING_SERVERURL: ""

        ingestor-server:
          envVars:
            # Image extraction settings
            APP_NVINGEST_STRUCTURED_ELEMENTS_MODALITY: ""
            APP_NVINGEST_IMAGE_ELEMENTS_MODALITY: "image"
            APP_NVINGEST_EXTRACTIMAGES: "True"

            # VLM embedding settings for ingestor
            APP_EMBEDDINGS_SERVERURL: "nemoretriever-vlm-embedding-ms:8000"
            APP_EMBEDDINGS_MODELNAME: "nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1"

        nv-ingest:
          envVars:
            EMBEDDING_NIM_ENDPOINT: "http://nemoretriever-vlm-embedding-ms:8000/v1"
            EMBEDDING_NIM_MODEL_NAME: "nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1"

    2. Deploy or upgrade the chart:

        helm upgrade --install rag -n rag https://helm.ngc.nvidia.com/nvidia/blueprint/charts/nvidia-blueprint-rag-v2.4.0-rc1.tgz \\
          --username '$oauthtoken' \\
          --password "${NGC_API_KEY}" \\
          --set imagePullSecret.password=$NGC_API_KEY \\
          --set ngcApiSecret.password=$NGC_API_KEY \\
          -f deploy/helm/nvidia-blueprint-rag/values.yaml

    3. Verify the VLM pods are running:

        kubectl get pods -n rag | grep -E "(vlm|embedding)"

        Expected pods:
        - rag-0 (VLM model deployment)
        - nemoretriever-vlm-embedding-ms (VLM embedding service)
"""

import asyncio
import base64
import json
import logging
import os
import time

import aiohttp

from ..base import BaseTestModule, TestStatus, test_case as itest_case
from ..utils.response_handlers import extract_streaming_text, print_response

logger = logging.getLogger(__name__)


class MultimodalQueryModule(BaseTestModule):
    """Multimodal Query integration tests"""

    _collection_name = "test_multimodal_query"

    # Candidate paths: data/multimodal is canonical (in repo); tests/data for CI (NGC dataset)
    _query_image_candidates = [
        "data/multimodal/query/Creme_clutch_purse1-small.jpg",
        "data/multimodal/Creme_clutch_purse1-small.jpg",
        "tests/data/query/Creme_clutch_purse1-small.jpg",
        "tests/data/Creme_clutch_purse1-small.jpg",
    ]
    _document_candidates = [
        "data/multimodal/product_catalog.pdf",
        "tests/data/product_catalog.pdf",
    ]

    def _resolve_path(self, candidates: list) -> str | None:
        """Resolve first existing path from candidates (repo-root relative)."""
        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..")
        )
        for rel_path in candidates:
            full_path = os.path.join(repo_root, rel_path)
            if os.path.exists(full_path):
                return full_path
        return None

    @property
    def _query_image_path(self) -> str | None:
        """Resolved path to query image."""
        return self._resolve_path(self._query_image_candidates)

    @property
    def _document_path(self) -> str | None:
        """Resolved path to document."""
        return self._resolve_path(self._document_candidates)

    def _get_base64_image(self, image_path: str) -> str:
        """Convert an image file to base64 encoding."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()

    def _create_multimodal_query(self, text_query: str, image_b64: str) -> list:
        """
        Create a multimodal query combining text and image.

        Args:
            text_query: The text portion of the query
            image_b64: Base64 encoded image

        Returns:
            List in OpenAI vision API format
        """
        image_input = f"data:image/jpeg;base64,{image_b64}"
        return [
            {"type": "text", "text": text_query},
            {
                "type": "image_url",
                "image_url": {
                    "url": image_input,
                    "detail": "auto"
                }
            }
        ]

    @itest_case(150, "Create Multimodal Query Test Collection")
    async def _test_create_collection(self) -> bool:
        """Create collection for multimodal query tests with 2048 embedding dimension."""
        logger.info("\n=== Test 150: Create Multimodal Query Test Collection ===")
        start = time.time()
        try:
            payload = {
                "collection_name": self._collection_name,
                "embedding_dimension": 2048,  # Multimodal embedding dimension
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ingestor_server_url}/v1/collection", json=payload
                ) as response:
                    result = await response.json()
                    ok = response.status == 200
                    self.add_test_result(
                        self._test_create_collection.test_number,
                        self._test_create_collection.test_name,
                        f"Create collection '{self._collection_name}' for multimodal query tests with embedding_dimension=2048 via POST /v1/collection.",
                        ["POST /v1/collection"],
                        ["collection_name", "embedding_dimension"],
                        time.time() - start,
                        TestStatus.SUCCESS if ok else TestStatus.FAILURE,
                        None if ok else f"API status {response.status}: {json.dumps(result, indent=2)}",
                    )
                    return ok
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            self.add_test_result(
                self._test_create_collection.test_number,
                self._test_create_collection.test_name,
                f"Create collection '{self._collection_name}' for multimodal query tests via POST /v1/collection.",
                ["POST /v1/collection"],
                ["collection_name", "embedding_dimension"],
                time.time() - start,
                TestStatus.FAILURE,
                f"Exception: {str(e)}",
            )
            return False

    @itest_case(151, "Ingest Product Catalog for Multimodal Query")
    async def _test_ingest_document(self) -> bool:
        """Ingest product catalog PDF for multimodal query testing."""
        logger.info("\n=== Test 151: Ingest Product Catalog for Multimodal Query ===")
        start = time.time()

        if not self._document_path:
            msg = f"Document file not found. Tried: {self._document_candidates}"
            logger.error(msg)
            self.add_test_result(
                self._test_ingest_document.test_number,
                self._test_ingest_document.test_name,
                f"Ingest product catalog using POST /v1/documents with blocking=false.",
                ["POST /v1/documents"],
                ["collection_name", "blocking"],
                time.time() - start,
                TestStatus.FAILURE,
                msg,
            )
            return False

        try:
            data = {
                "collection_name": self._collection_name,
                "blocking": False,
                "split_options": {"chunk_size": 512, "chunk_overlap": 150},
                "custom_metadata": [],
                "generate_summary": False,
            }
            form_data = aiohttp.FormData()
            with open(self._document_path, "rb") as f:
                file_content = f.read()
            form_data.add_field(
                "documents",
                file_content,
                filename=os.path.basename(self._document_path),
                content_type="application/pdf",
            )
            form_data.add_field("data", json.dumps(data), content_type="application/json")

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ingestor_server_url}/v1/documents", data=form_data
                ) as response:
                    result = await response.json()
                    if response.status != 200:
                        self.add_test_result(
                            self._test_ingest_document.test_number,
                            self._test_ingest_document.test_name,
                            f"Ingest product catalog using POST /v1/documents with blocking=false.",
                            ["POST /v1/documents"],
                            ["collection_name", "blocking"],
                            time.time() - start,
                            TestStatus.FAILURE,
                            f"API status {response.status}: {json.dumps(result, indent=2)}",
                        )
                        return False

                    task_id = result.get("task_id")
                    if not task_id:
                        self.add_test_result(
                            self._test_ingest_document.test_number,
                            self._test_ingest_document.test_name,
                            f"Ingest product catalog using POST /v1/documents with blocking=false.",
                            ["POST /v1/documents"],
                            ["collection_name", "blocking"],
                            time.time() - start,
                            TestStatus.FAILURE,
                            "No task_id returned",
                        )
                        return False

                    self.multimodal_query_task_id = task_id
                    self.add_test_result(
                        self._test_ingest_document.test_number,
                        self._test_ingest_document.test_name,
                        f"Ingest product catalog using POST /v1/documents with blocking=false. Task ID: {task_id}",
                        ["POST /v1/documents"],
                        ["collection_name", "blocking"],
                        time.time() - start,
                        TestStatus.SUCCESS,
                    )
                    return True
        except Exception as e:
            logger.error(f"Error ingesting document: {e}")
            self.add_test_result(
                self._test_ingest_document.test_number,
                self._test_ingest_document.test_name,
                f"Ingest product catalog using POST /v1/documents with blocking=false.",
                ["POST /v1/documents"],
                ["collection_name", "blocking"],
                time.time() - start,
                TestStatus.FAILURE,
                f"Exception: {str(e)}",
            )
            return False

    @itest_case(152, "Wait for Multimodal Document Ingestion")
    async def _test_wait_for_ingestion(self) -> bool:
        """Wait for document ingestion to complete."""
        logger.info("\n=== Test 152: Wait for Multimodal Document Ingestion ===")
        start = time.time()
        try:
            task_id = getattr(self, "multimodal_query_task_id", None)
            if not task_id:
                self.add_test_result(
                    self._test_wait_for_ingestion.test_number,
                    self._test_wait_for_ingestion.test_name,
                    f"Wait for ingestion completion using GET /v1/status.",
                    ["GET /v1/status"],
                    ["task_id"],
                    time.time() - start,
                    TestStatus.FAILURE,
                    "No task_id from previous step",
                )
                return False

            timeout = 120  # 2 minutes timeout for PDF processing
            poll_interval = 3
            t0 = time.time()
            while time.time() - t0 < timeout:
                async with aiohttp.ClientSession() as session:
                    params = {"task_id": task_id}
                    async with session.get(
                        f"{self.ingestor_server_url}/v1/status", params=params
                    ) as response:
                        result = await response.json()
                        if response.status != 200:
                            self.add_test_result(
                                self._test_wait_for_ingestion.test_number,
                                self._test_wait_for_ingestion.test_name,
                                f"Wait for ingestion completion using GET /v1/status.",
                                ["GET /v1/status"],
                                ["task_id"],
                                time.time() - start,
                                TestStatus.FAILURE,
                                f"API status {response.status}: {json.dumps(result, indent=2)}",
                            )
                            return False
                        state = result.get("state")
                        if state == "FINISHED":
                            self.add_test_result(
                                self._test_wait_for_ingestion.test_number,
                                self._test_wait_for_ingestion.test_name,
                                f"Wait for ingestion completion using GET /v1/status. Completed in {time.time() - t0:.1f}s",
                                ["GET /v1/status"],
                                ["task_id"],
                                time.time() - start,
                                TestStatus.SUCCESS,
                            )
                            return True
                        if state == "FAILED":
                            self.add_test_result(
                                self._test_wait_for_ingestion.test_number,
                                self._test_wait_for_ingestion.test_name,
                                f"Wait for ingestion completion using GET /v1/status.",
                                ["GET /v1/status"],
                                ["task_id"],
                                time.time() - start,
                                TestStatus.FAILURE,
                                f"Task {task_id} failed: {json.dumps(result, indent=2)}",
                            )
                            return False
                        logger.info(f"Ingestion state: {state}, waiting...")
                await asyncio.sleep(poll_interval)

            self.add_test_result(
                self._test_wait_for_ingestion.test_number,
                self._test_wait_for_ingestion.test_name,
                f"Wait for ingestion completion using GET /v1/status.",
                ["GET /v1/status"],
                ["task_id"],
                time.time() - start,
                TestStatus.FAILURE,
                f"Task {task_id} timed out after {timeout}s",
            )
            return False
        except Exception as e:
            logger.error(f"Error waiting for ingestion: {e}")
            self.add_test_result(
                self._test_wait_for_ingestion.test_number,
                self._test_wait_for_ingestion.test_name,
                f"Wait for ingestion completion using GET /v1/status.",
                ["GET /v1/status"],
                ["task_id"],
                time.time() - start,
                TestStatus.FAILURE,
                f"Exception: {str(e)}",
            )
            return False

    @itest_case(153, "Multimodal Search with Image Query")
    async def _test_multimodal_search(self) -> bool:
        """Test search API with multimodal query (text + image)."""
        logger.info("\n=== Test 153: Multimodal Search with Image Query ===")
        start = time.time()

        # Check if query image exists
        if not self._query_image_path:
            msg = f"Query image not found. Tried: {self._query_image_candidates}"
            logger.error(msg)
            self.add_test_result(
                self._test_multimodal_search.test_number,
                self._test_multimodal_search.test_name,
                "Test search with multimodal query (text + image).",
                ["POST /v1/search"],
                ["query", "collection_names", "vdb_top_k"],
                time.time() - start,
                TestStatus.FAILURE,
                msg,
            )
            return False

        try:
            # Create multimodal query - search for purse details
            image_b64 = self._get_base64_image(self._query_image_path)
            multimodal_query = self._create_multimodal_query(
                "Tell me about this purse",
                image_b64
            )

            payload = {
                "query": multimodal_query,
                "messages": [],
                "use_knowledge_base": True,
                "collection_names": [self._collection_name],
                "vdb_top_k": 10,
                "reranker_top_k": 5,
                "enable_reranker": False,
                "filter_expr": "",
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.rag_server_url}/v1/search",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    result = await print_response(response)

                    if response.status != 200:
                        self.add_test_result(
                            self._test_multimodal_search.test_number,
                            self._test_multimodal_search.test_name,
                            "Test search with multimodal query (text + image).",
                            ["POST /v1/search"],
                            ["query", "collection_names", "vdb_top_k"],
                            time.time() - start,
                            TestStatus.FAILURE,
                            f"API status {response.status}: {json.dumps(result, indent=2)}",
                        )
                        return False

                    # Verify search results
                    results = result.get("results", [])
                    if not results:
                        self.add_test_result(
                            self._test_multimodal_search.test_number,
                            self._test_multimodal_search.test_name,
                            "Test search with multimodal query (text + image).",
                            ["POST /v1/search"],
                            ["query", "collection_names", "vdb_top_k"],
                            time.time() - start,
                            TestStatus.FAILURE,
                            "No search results returned",
                        )
                        return False

                    # Log result types and counts
                    image_results = sum(1 for r in results if r.get("document_type") == "image")
                    text_results = sum(1 for r in results if r.get("document_type") != "image")
                    logger.info(f"Search returned {len(results)} results: {image_results} images, {text_results} text chunks")

                    # Verify expected content in search results
                    expected_phrase = "offering a timeless appeal with its soft silhouette"
                    all_content = " ".join([r.get("content", "") for r in results if r.get("document_type") != "image"])
                    content_found = expected_phrase.lower() in all_content.lower()

                    if not content_found:
                        logger.warning(f"Expected phrase not found in search results: '{expected_phrase}'")
                        logger.info(f"Search results content preview: {all_content[:500]}...")
                        self.add_test_result(
                            self._test_multimodal_search.test_number,
                            self._test_multimodal_search.test_name,
                            "Test search with multimodal query (text + image). Content validation failed.",
                            ["POST /v1/search"],
                            ["query", "collection_names", "vdb_top_k"],
                            time.time() - start,
                            TestStatus.FAILURE,
                            f"Expected phrase '{expected_phrase}' not found in search results",
                        )
                        return False

                    logger.info(f"✅ Content validation passed: found expected phrase in search results")

                    self.add_test_result(
                        self._test_multimodal_search.test_number,
                        self._test_multimodal_search.test_name,
                        f"Test search with multimodal query (text + image). Found {len(results)} results ({image_results} images, {text_results} text). Content validated.",
                        ["POST /v1/search"],
                        ["query", "collection_names", "vdb_top_k"],
                        time.time() - start,
                        TestStatus.SUCCESS,
                    )
                    return True

        except Exception as e:
            logger.error(f"Error in multimodal search: {e}")
            self.add_test_result(
                self._test_multimodal_search.test_number,
                self._test_multimodal_search.test_name,
                "Test search with multimodal query (text + image).",
                ["POST /v1/search"],
                ["query", "collection_names", "vdb_top_k"],
                time.time() - start,
                TestStatus.FAILURE,
                f"Exception: {str(e)}",
            )
            return False

    @itest_case(154, "Multimodal Generate with Image Query")
    async def _test_multimodal_generate(self) -> bool:
        """Test generate API with multimodal query (text + image)."""
        logger.info("\n=== Test 154: Multimodal Generate with Image Query ===")
        start = time.time()

        # Check if query image exists
        if not self._query_image_path:
            msg = f"Query image not found. Tried: {self._query_image_candidates}"
            logger.error(msg)
            self.add_test_result(
                self._test_multimodal_generate.test_number,
                self._test_multimodal_generate.test_name,
                "Test generate with multimodal query (text + image) using VLM.",
                ["POST /v1/generate"],
                ["messages", "collection_names", "enable_vlm_inference"],
                time.time() - start,
                TestStatus.FAILURE,
                msg,
            )
            return False

        try:
            # Create multimodal query - ask about price to validate 69.9 in response
            image_b64 = self._get_base64_image(self._query_image_path)
            multimodal_query = self._create_multimodal_query(
                "How much does this purse cost?",
                image_b64
            )

            messages = [
                {
                    "role": "user",
                    "content": multimodal_query
                }
            ]

            payload = {
                "messages": messages,
                "use_knowledge_base": True,
                "temperature": 0.2,
                "top_p": 0.7,
                "max_tokens": 1024,
                "reranker_top_k": 2,
                "vdb_top_k": 10,
                "collection_names": [self._collection_name],
                "enable_query_rewriting": True,
                "enable_citations": True,
                "enable_vlm_inference": True,
                "stop": [],
                "filter_expr": "",
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.rag_server_url}/v1/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=180)
                ) as response:
                    result = await print_response(response)

                    if response.status != 200:
                        self.add_test_result(
                            self._test_multimodal_generate.test_number,
                            self._test_multimodal_generate.test_name,
                            "Test generate with multimodal query (text + image) using VLM.",
                            ["POST /v1/generate"],
                            ["messages", "collection_names", "enable_vlm_inference"],
                            time.time() - start,
                            TestStatus.FAILURE,
                            f"API status {response.status}: {json.dumps(result, indent=2)}",
                        )
                        return False

                    # Extract response content
                    content = ""
                    if result.get("streaming_response"):
                        content = extract_streaming_text(result)
                    elif result.get("choices"):
                        content = result["choices"][0].get("message", {}).get("content", "")

                    if not content:
                        self.add_test_result(
                            self._test_multimodal_generate.test_number,
                            self._test_multimodal_generate.test_name,
                            "Test generate with multimodal query (text + image) using VLM.",
                            ["POST /v1/generate"],
                            ["messages", "collection_names", "enable_vlm_inference"],
                            time.time() - start,
                            TestStatus.FAILURE,
                            "No content in response",
                        )
                        return False

                    logger.info(f"Generated response preview: {content[:500]}...")

                    # Verify expected price in response (69.9 for the Creme clutch purse)
                    expected_price = "69.9"
                    price_found = expected_price in content

                    if not price_found:
                        logger.warning(f"Expected price '{expected_price}' not found in response")
                        logger.info(f"Full response: {content}")
                        self.add_test_result(
                            self._test_multimodal_generate.test_number,
                            self._test_multimodal_generate.test_name,
                            "Test generate with multimodal query (text + image) using VLM. Price validation failed.",
                            ["POST /v1/generate"],
                            ["messages", "collection_names", "enable_vlm_inference"],
                            time.time() - start,
                            TestStatus.FAILURE,
                            f"Expected price '{expected_price}' not found in generated response",
                        )
                        return False

                    logger.info(f"✅ Price validation passed: found '{expected_price}' in response")

                    # Check for citations
                    citations_count = 0
                    if "citations" in result:
                        citations = result["citations"]
                        citations_count = len(citations.get("results", []))
                        logger.info(f"Response includes {citations_count} citations")

                    self.add_test_result(
                        self._test_multimodal_generate.test_number,
                        self._test_multimodal_generate.test_name,
                        f"Test generate with multimodal query (text + image) using VLM. Response length: {len(content)} chars, {citations_count} citations. Price validated.",
                        ["POST /v1/generate"],
                        ["messages", "collection_names", "enable_vlm_inference"],
                        time.time() - start,
                        TestStatus.SUCCESS,
                    )
                    return True

        except Exception as e:
            logger.error(f"Error in multimodal generate: {e}")
            self.add_test_result(
                self._test_multimodal_generate.test_number,
                self._test_multimodal_generate.test_name,
                "Test generate with multimodal query (text + image) using VLM.",
                ["POST /v1/generate"],
                ["messages", "collection_names", "enable_vlm_inference"],
                time.time() - start,
                TestStatus.FAILURE,
                f"Exception: {str(e)}",
            )
            return False

    @itest_case(155, "Delete Multimodal Query Test Collection")
    async def _test_delete_collection(self) -> bool:
        """Delete the multimodal query test collection."""
        logger.info("\n=== Test 155: Delete Multimodal Query Test Collection ===")
        start = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.delete(
                    f"{self.ingestor_server_url}/v1/collections",
                    json=[self._collection_name],
                ) as response:
                    result = await response.json()
                    ok = response.status == 200
                    self.add_test_result(
                        self._test_delete_collection.test_number,
                        self._test_delete_collection.test_name,
                        f"Delete collection '{self._collection_name}' via DELETE /v1/collections.",
                        ["DELETE /v1/collections"],
                        ["collection_names"],
                        time.time() - start,
                        TestStatus.SUCCESS if ok else TestStatus.FAILURE,
                        None if ok else f"API status {response.status}: {json.dumps(result, indent=2)}",
                    )
                    return ok
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            self.add_test_result(
                self._test_delete_collection.test_number,
                self._test_delete_collection.test_name,
                f"Delete collection '{self._collection_name}' via DELETE /v1/collections.",
                ["DELETE /v1/collections"],
                ["collection_names"],
                time.time() - start,
                TestStatus.FAILURE,
                f"Exception: {str(e)}",
            )
            return False

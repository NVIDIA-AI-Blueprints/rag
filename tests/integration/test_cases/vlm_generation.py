# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
VLM generation integration tests
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


class VLMGenerationModule(BaseTestModule):
    """VLM generation integration tests"""

    _collection_name = "test_vlm_generation"

    @itest_case(61, "Create VLM Test Collection")
    async def _test_create_collection(self) -> bool:
        logger.info("\n=== Test 61: Create VLM Test Collection ===")
        start = time.time()
        try:
            payload = {
                "collection_name": self._collection_name,
                "embedding_dimension": 2048,
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.ingestor_server_url}/v1/collection", json=payload) as response:
                    result = await response.json()
                    ok = response.status == 200
                    self.add_test_result(
                        self._test_create_collection.test_number,
                        self._test_create_collection.test_name,
                        f"Create collection '{self._collection_name}' for VLM tests via POST /v1/collection.",
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
                f"Create collection '{self._collection_name}' for VLM tests via POST /v1/collection.",
                ["POST /v1/collection"],
                ["collection_name", "embedding_dimension"],
                time.time() - start,
                TestStatus.FAILURE,
                f"Exception: {str(e)}",
            )
            return False

    @itest_case(62, "Ingest VLM Test Image (citations)")
    async def _test_ingest_image(self) -> bool:
        logger.info("\n=== Test 62: Ingest VLM Test Image ===")
        start = time.time()

        image_file = "tests/data/stock_prices.jpg"
        if not os.path.exists(image_file):
            msg = f"Image file not found: {image_file}"
            logger.error(msg)
            self.add_test_result(
                self._test_ingest_image.test_number,
                self._test_ingest_image.test_name,
                f"Ingest test image using POST /v1/documents with blocking=false.",
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
            with open(image_file, "rb") as f:
                file_content = f.read()
            form_data.add_field(
                "documents",
                file_content,
                filename=os.path.basename(image_file),
                content_type="image/png",
            )
            form_data.add_field("data", json.dumps(data), content_type="application/json")

            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.ingestor_server_url}/v1/documents", data=form_data) as response:
                    result = await response.json()
                    if response.status != 200:
                        self.add_test_result(
                            self._test_ingest_image.test_number,
                            self._test_ingest_image.test_name,
                            f"Ingest test image using POST /v1/documents with blocking=false.",
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
                            self._test_ingest_image.test_number,
                            self._test_ingest_image.test_name,
                            f"Ingest test image using POST /v1/documents with blocking=false.",
                            ["POST /v1/documents"],
                            ["collection_name", "blocking"],
                            time.time() - start,
                            TestStatus.FAILURE,
                            "No task_id returned",
                        )
                        return False

                    self.vlm_task_id = task_id
                    self.add_test_result(
                        self._test_ingest_image.test_number,
                        self._test_ingest_image.test_name,
                        f"Ingest test image using POST /v1/documents with blocking=false.",
                        ["POST /v1/documents"],
                        ["collection_name", "blocking"],
                        time.time() - start,
                        TestStatus.SUCCESS,
                    )
                    return True
        except Exception as e:
            logger.error(f"Error ingesting image: {e}")
            self.add_test_result(
                self._test_ingest_image.test_number,
                self._test_ingest_image.test_name,
                f"Ingest test image using POST /v1/documents with blocking=false.",
                ["POST /v1/documents"],
                ["collection_name", "blocking"],
                time.time() - start,
                TestStatus.FAILURE,
                f"Exception: {str(e)}",
            )
            return False

    @itest_case(63, "Wait for VLM Image Ingestion")
    async def _test_wait_for_ingestion(self) -> bool:
        logger.info("\n=== Test 63: Wait for VLM Image Ingestion ===")
        start = time.time()
        try:
            task_id = getattr(self, "vlm_task_id", None)
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

            timeout = 60
            poll_interval = 2
            t0 = time.time()
            while time.time() - t0 < timeout:
                async with aiohttp.ClientSession() as session:
                    params = {"task_id": task_id}
                    async with session.get(f"{self.ingestor_server_url}/v1/status", params=params) as response:
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
                                f"Wait for ingestion completion using GET /v1/status.",
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
                                f"Task {task_id} failed",
                            )
                            return False
                await asyncio.sleep(poll_interval)

            self.add_test_result(
                self._test_wait_for_ingestion.test_number,
                self._test_wait_for_ingestion.test_name,
                f"Wait for ingestion completion using GET /v1/status.",
                ["GET /v1/status"],
                ["task_id"],
                time.time() - start,
                TestStatus.FAILURE,
                f"Task {task_id} timed out",
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

    @itest_case(64, "VLM Generation via Citations")
    async def _test_vlm_generation_via_citations(self) -> bool:
        logger.info("\n=== Test 64: VLM Generation via Citations ===")
        start = time.time()
        payload = {
            "messages": [{"role": "user", "content": "What color is Wipro IT Companies represented in?"}],
            "use_knowledge_base": True,
            "temperature": 0,
            "top_p": 0.1,
            "max_tokens": 512,
            "reranker_top_k": 3,
            "vdb_top_k": 10,
            "collection_names": [self._collection_name],
            "enable_query_rewriting": False,
            "enable_reranker": True,
            "enable_citations": True,
            "enable_vlm_inference": True,
            "stop": [],
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.rag_server_url}/v1/generate", json=payload) as response:
                    result = await print_response(response)
                    if response.status != 200:
                        self.add_test_result(
                            self._test_vlm_generation_via_citations.test_number,
                            self._test_vlm_generation_via_citations.test_name,
                            "Generate with image citations to trigger VLM.",
                            ["POST /v1/generate"],
                            ["messages", "collection_names"],
                            time.time() - start,
                            TestStatus.FAILURE,
                            f"API status {response.status}: {json.dumps(result, indent=2)}",
                        )
                        return False

                    content = ""
                    if result.get("streaming_response"):
                        content = extract_streaming_text(result)
                    elif result.get("choices"):
                        content = result["choices"][0].get("message", {}).get("content", "")

                    # Expect the known keywords from the sample image
                    ok = "red" in content.lower()
                    self.add_test_result(
                        self._test_vlm_generation_via_citations.test_number,
                        self._test_vlm_generation_via_citations.test_name,
                        "Generate with image citations to trigger VLM.",
                        ["POST /v1/generate"],
                        ["messages", "collection_names"],
                        time.time() - start,
                        TestStatus.SUCCESS if ok else TestStatus.FAILURE,
                        None if ok else f"Unexpected content: {content}",
                    )
                    return ok
        except Exception as e:
            logger.error(f"Error in VLM generation via citations: {e}")
            self.add_test_result(
                self._test_vlm_generation_via_citations.test_number,
                self._test_vlm_generation_via_citations.test_name,
                "Generate with image citations to trigger VLM.",
                ["POST /v1/generate"],
                ["messages", "collection_names"],
                time.time() - start,
                TestStatus.FAILURE,
                f"Exception: {str(e)}",
            )
            return False

    @itest_case(65, "Delete VLM Test Collection")
    async def _test_delete_collection(self) -> bool:
        logger.info("\n=== Test 65: Delete VLM Test Collection ===")
        start = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.delete(
                    f"{self.ingestor_server_url}/v1/collections", json=[self._collection_name]
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



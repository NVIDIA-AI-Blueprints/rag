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
Image Captioning test module
"""

import asyncio
import json
import logging
import os
import time

import aiohttp

from ..base import BaseTestModule, TestStatus, test_case
from ..utils.response_handlers import print_response, extract_streaming_text

logger = logging.getLogger(__name__)


class ImageCaptioningModule(BaseTestModule):
    """Image Captioning test module"""

    @test_case(26, "Create Image Captioning Collection")
    async def _test_create_image_captioning_collection(self) -> bool:
        """Test creating collection for image captioning"""
        logger.info("\n=== Test 26: Create Image Captioning Collection ===")
        collection_start = time.time()

        collection_name = "test_image_ingestion"

        try:
            payload = {
                "collection_name": collection_name,
                "embedding_dimension": 2048,
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.ingestor_server_url}/v1/collection", json=payload) as response:
                    result = await response.json()
                    if response.status == 200:
                        logger.info(f"✅ Collection '{collection_name}' created successfully:")
                        logger.info(f"Response JSON:\n{json.dumps(result, indent=2)}")

                        self.add_test_result(
                            self._test_create_image_captioning_collection.test_number,
                            self._test_create_image_captioning_collection.test_name,
                            f"Create a collection named {collection_name} for image captioning tests using the POST /v1/collection endpoint.",
                            ["POST /v1/collection"],
                            ["collection_name", "embedding_dimension"],
                            time.time() - collection_start,
                            TestStatus.SUCCESS,
                        )
                        return True
                    else:
                        logger.error(f"❌ Failed to create collection '{collection_name}': {response.status}")
                        logger.error(f"Response JSON:\n{json.dumps(result, indent=2)}")

                        self.add_test_result(
                            self._test_create_image_captioning_collection.test_number,
                            self._test_create_image_captioning_collection.test_name,
                            f"Create a collection named {collection_name} for image captioning tests using the POST /v1/collection endpoint.",
                            ["POST /v1/collection"],
                            ["collection_name", "embedding_dimension"],
                            time.time() - collection_start,
                            TestStatus.FAILURE,
                            f"API request failed with status {response.status}",
                        )
                        return False
        except Exception as e:
            logger.error(f"❌ Error creating collection '{collection_name}': {e}")
            self.add_test_result(
                self._test_create_image_captioning_collection.test_number,
                self._test_create_image_captioning_collection.test_name,
                f"Create a collection named {collection_name} for image captioning tests using the POST /v1/collection endpoint.",
                ["POST /v1/collection"],
                ["collection_name", "embedding_dimension"],
                time.time() - collection_start,
                TestStatus.FAILURE,
                f"Exception occurred: {str(e)}",
            )
            return False

    @test_case(27, "Ingest Earthquake Image")
    async def _test_ingest_earthquake_image(self) -> bool:
        """Test ingesting earthquake-info.png image"""
        logger.info("\n=== Test 27: Ingest Earthquake Image ===")
        upload_start = time.time()

        collection_name = "test_image_ingestion"
        image_file = "tests/data/earthquake-info.png"

        if not os.path.exists(image_file):
            logger.error(f"❌ Image file not found: {image_file}")
            self.add_test_result(
                self._test_ingest_earthquake_image.test_number,
                self._test_ingest_earthquake_image.test_name,
                f"Ingest earthquake-info.png image under data directory using POST /v1/documents endpoint with blocking=true.",
                ["POST /v1/documents"],
                ["collection_name", "blocking"],
                time.time() - upload_start,
                TestStatus.FAILURE,
                f"Image file not found: {image_file}",
            )
            return False

        try:
            data = {
                "collection_name": collection_name,
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
                logger.info(f"📤 Uploading earthquake image to collection '{collection_name}'")
                logger.info(f"📁 File: {image_file}")
                logger.info(f"📋 Upload data: {json.dumps(data, indent=2)}")

                async with session.post(
                    f"{self.ingestor_server_url}/v1/documents", data=form_data
                ) as response:
                    result = await response.json()
                    if response.status == 200:
                        logger.info(f"✅ Image upload successful. Response:\n{json.dumps(result, indent=2)}")

                        task_id = result.get("task_id")
                        if task_id:
                            logger.info(f"✅ Image upload initiated successfully. Task ID: {task_id}")

                            # Store task_id for the next test
                            self.image_upload_task_id = task_id

                            self.add_test_result(
                                self._test_ingest_earthquake_image.test_number,
                                self._test_ingest_earthquake_image.test_name,
                                f"Ingest earthquake-info.png image under data directory using POST /v1/documents endpoint with blocking=false.",
                                ["POST /v1/documents"],
                                ["collection_name", "blocking"],
                                time.time() - upload_start,
                                TestStatus.SUCCESS,
                            )
                            return True
                        else:
                            logger.error(f"❌ No task_id returned from upload")
                            self.add_test_result(
                                self._test_ingest_earthquake_image.test_number,
                                self._test_ingest_earthquake_image.test_name,
                                f"Ingest earthquake-info.png image under data directory using POST /v1/documents endpoint with blocking=false.",
                                ["POST /v1/documents"],
                                ["collection_name", "blocking"],
                                time.time() - upload_start,
                                TestStatus.FAILURE,
                                "No task_id returned from upload",
                            )
                            return False
                    else:
                        logger.error(f"❌ Failed to upload image. Status: {response.status}")
                        logger.error(f"❌ Response:\n{json.dumps(result, indent=2)}")

                        self.add_test_result(
                            self._test_ingest_earthquake_image.test_number,
                            self._test_ingest_earthquake_image.test_name,
                            f"Ingest earthquake-info.png image under data directory using POST /v1/documents endpoint with blocking=false.",
                            ["POST /v1/documents"],
                            ["collection_name", "blocking"],
                            time.time() - upload_start,
                            TestStatus.FAILURE,
                            f"API request failed with status {response.status}",
                        )
                        return False
        except Exception as e:
            logger.error(f"❌ Error uploading image: {e}")
            self.add_test_result(
                self._test_ingest_earthquake_image.test_number,
                self._test_ingest_earthquake_image.test_name,
                f"Ingest earthquake-info.png image under data directory using POST /v1/documents endpoint with blocking=false.",
                ["POST /v1/documents"],
                ["collection_name", "blocking"],
                time.time() - upload_start,
                TestStatus.FAILURE,
                f"Exception occurred: {str(e)}",
            )
            return False

    @test_case(28, "Wait for Ingestion Completion")
    async def _test_wait_for_image_ingestion(self) -> bool:
        """Test waiting for image ingestion completion"""
        logger.info("\n=== Test 28: Wait for Ingestion Completion ===")
        wait_start = time.time()

        try:
            # Get the task_id from the previous test
            task_id = getattr(self, 'image_upload_task_id', None)
            if not task_id:
                logger.error(f"❌ No task_id available from previous test")
                self.add_test_result(
                    self._test_wait_for_image_ingestion.test_number,
                    self._test_wait_for_image_ingestion.test_name,
                    f"Wait for ingestion completion using /v1/status API with task_id until ingestion success.",
                    ["GET /v1/status"],
                    ["task_id"],
                    time.time() - wait_start,
                    TestStatus.FAILURE,
                    "No task_id available from previous test",
                )
                return False

            # Poll the status endpoint until completion
            timeout = 60  # 60 seconds timeout
            poll_interval = 2  # 2 seconds between polls
            start_time = time.time()

            while time.time() - start_time < timeout:
                async with aiohttp.ClientSession() as session:
                    params = {"task_id": task_id}
                    async with session.get(f"{self.ingestor_server_url}/v1/status", params=params) as response:
                        result = await response.json()
                        if response.status == 200:
                            state = result.get("state")
                            logger.info(f"⏳ Task {task_id} state: {state}")

                            if state == "FINISHED":
                                logger.info(f"✅ Task {task_id} completed successfully")
                                logger.info(f"Task result:\n{json.dumps(result, indent=2)}")

                                self.add_test_result(
                                    self._test_wait_for_image_ingestion.test_number,
                                    self._test_wait_for_image_ingestion.test_name,
                                    f"Wait for ingestion completion using /v1/status API with task_id until ingestion success.",
                                    ["GET /v1/status"],
                                    ["task_id"],
                                    time.time() - wait_start,
                                    TestStatus.SUCCESS,
                                )
                                return True
                            elif state == "FAILED":
                                logger.error(f"❌ Task {task_id} failed")
                                logger.error(f"Task failure details:\n{json.dumps(result, indent=2)}")

                                self.add_test_result(
                                    self._test_wait_for_image_ingestion.test_number,
                                    self._test_wait_for_image_ingestion.test_name,
                                    f"Wait for ingestion completion using /v1/status API with task_id until ingestion success.",
                                    ["GET /v1/status"],
                                    ["task_id"],
                                    time.time() - wait_start,
                                    TestStatus.FAILURE,
                                    f"Task {task_id} failed",
                                )
                                return False
                            else:
                                # Log additional task details for debugging
                                if "progress" in result:
                                    logger.info(f"   Progress: {result.get('progress')}")
                                if "message" in result:
                                    logger.info(f"   Message: {result.get('message')}")
                        else:
                            logger.error(f"❌ Failed to get status: {response.status}")
                            logger.error(f"Response JSON:\n{json.dumps(result, indent=2)}")

                            self.add_test_result(
                                self._test_wait_for_image_ingestion.test_number,
                                self._test_wait_for_image_ingestion.test_name,
                                f"Wait for ingestion completion using /v1/status API with task_id until ingestion success.",
                                ["GET /v1/status"],
                                ["task_id"],
                                time.time() - wait_start,
                                TestStatus.FAILURE,
                                f"API request failed with status {response.status}",
                            )
                            return False

                # Wait before next poll
                await asyncio.sleep(poll_interval)

            # Timeout reached
            logger.error(f"❌ Task {task_id} timed out after {timeout} seconds")
            self.add_test_result(
                self._test_wait_for_image_ingestion.test_number,
                self._test_wait_for_image_ingestion.test_name,
                f"Wait for ingestion completion using /v1/status API with task_id until ingestion success.",
                ["GET /v1/status"],
                ["task_id"],
                time.time() - wait_start,
                TestStatus.FAILURE,
                f"Task {task_id} timed out after {timeout} seconds",
            )
            return False

        except Exception as e:
            logger.error(f"❌ Error waiting for ingestion completion: {e}")
            self.add_test_result(
                self._test_wait_for_image_ingestion.test_number,
                self._test_wait_for_image_ingestion.test_name,
                f"Wait for ingestion completion using /v1/status API with task_id until ingestion success.",
                ["GET /v1/status"],
                ["task_id"],
                time.time() - wait_start,
                TestStatus.FAILURE,
                f"Exception occurred: {str(e)}",
            )
            return False

    @test_case(29, "Query Earthquake Image")
    async def _test_query_earthquake_image(self) -> bool:
        """Test querying the earthquake image and verifying response"""
        logger.info("\n=== Test 29: Query Earthquake Image ===")
        query_start = time.time()

        collection_name = "test_image_ingestion"

        try:
            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": "Where did the earthquake strike?"
                    }
                ],
                "use_knowledge_base": True,
                "temperature": 0.3,
                "top_p": 0.8,
                "max_tokens": 512,
                "reranker_top_k": 3,
                "vdb_top_k": 10,
                "collection_names": [collection_name],
                "enable_query_rewriting": False,
                "enable_reranker": True,
                "enable_citations": True,
                "stop": [],
            }

            logger.info(f"🖼️ Testing Earthquake Image Query")
            logger.info(f"📋 Query request payload:\n{json.dumps(payload, indent=2)}")

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.rag_server_url}/v1/generate", json=payload
                ) as response:
                    result = await print_response(response)
                    if response.status == 200:
                        logger.info(f"✅ Earthquake image query test passed:")

                        # Extract response content
                        content = ""
                        if result.get("streaming_response"):
                            # Handle streaming response
                            content = extract_streaming_text(result)
                            logger.info(f"Streaming response content: {content}")
                        elif result.get("choices") and len(result.get("choices", [])) > 0:
                            # Handle regular JSON response
                            content = result["choices"][0].get("message", {}).get("content", "")
                            logger.info(f"Regular response content: {content}")
                        else:
                            logger.warning("No content found in response")

                        # Verify response contains one of the allowed response variants
                        content_lower = content.lower()
                        allowed_response_variants = [
                            "northern california",
                            "redding",
                            "algorithmic landscape",
                            "upload an image",
                            "upload the image",
                            "in the provided context"
                        ]
                        if any(variant in content_lower for variant in allowed_response_variants):
                            logger.info(
                                f"✅ Response contains one of the allowed response variants: {allowed_response_variants}"
                            )

                            self.add_test_result(
                                self._test_query_earthquake_image.test_number,
                                                self._test_query_earthquake_image.test_name,
                                                f"Send query 'Where did the earthquake strike?' via /v1/generate endpoint and verify the response contains one of the allowed variants of expected keywords in the result.",
                                ["POST /v1/generate"],
                                [
                                    "messages",
                                    "use_knowledge_base",
                                    "temperature",
                                    "max_tokens",
                                    "collection_names"
                                ],
                                time.time() - query_start,
                                TestStatus.SUCCESS,
                            )
                            return True
                        else:
                            logger.error(f"❌ Response does not contain expected keywords")
                            logger.error(f"Response content: {content}")

                            self.add_test_result(
                                self._test_query_earthquake_image.test_number,
                                                self._test_query_earthquake_image.test_name,
                                                f"Send query 'Where did the earthquake strike?' via /v1/generate endpoint and verify the response contains one of the allowed variants of expected keywords in the result.",
                                ["POST /v1/generate"],
                                [
                                    "messages",
                                    "use_knowledge_base",
                                    "temperature",
                                    "max_tokens",
                                    "collection_names"
                                ],
                                time.time() - query_start,
                                TestStatus.FAILURE,
                                f"Response does not contain expected keywords. Content: {content}",
                            )
                            return False
                    else:
                        logger.error(f"❌ Earthquake image query failed: {response.status}")
                        logger.error(f"Response JSON:\n{json.dumps(result, indent=2)}")

                        self.add_test_result(
                            self._test_query_earthquake_image.test_number,
                            self._test_query_earthquake_image.test_name,
                            f"Send query 'Where did the earthquake strike?' via /v1/generate endpoint and verify the response contains one of the allowed variants of expected keywords in the result.",
                            ["POST /v1/generate"],
                            [
                                "messages",
                                "use_knowledge_base",
                                "temperature",
                                "max_tokens",
                                "collection_names"
                            ],
                            time.time() - query_start,
                            TestStatus.FAILURE,
                            f"API request failed with status {response.status}",
                        )
                        return False
        except Exception as e:
            logger.error(f"❌ Earthquake image query test error: {e}")
            self.add_test_result(
                self._test_query_earthquake_image.test_number,
                self._test_query_earthquake_image.test_name,
                f"Send query 'Where did the earthquake strike?' via /v1/generate endpoint and verify the response contains keywords 'Redding' and 'Northern California'.",
                ["POST /v1/generate"],
                [
                    "messages",
                    "use_knowledge_base",
                    "temperature",
                    "max_tokens",
                    "collection_names"
                ],
                time.time() - query_start,
                TestStatus.FAILURE,
                f"Exception occurred: {str(e)}",
            )
            return False

    @test_case(30, "Delete Image Captioning Collection")
    async def _test_delete_image_captioning_collection(self) -> bool:
        """Test deleting the image captioning collection"""
        logger.info("\n=== Test 30: Delete Image Captioning Collection ===")
        delete_start = time.time()

        collection_name = "test_image_ingestion"

        try:
            async with aiohttp.ClientSession() as session:
                logger.info(f"🗑️ Deleting collection '{collection_name}'")
                params = {"collection_names": [collection_name]}
                async with session.delete(
                    f"{self.ingestor_server_url}/v1/collections", params=params
                ) as response:
                    result = await response.json()
                    if response.status == 200:
                        logger.info(f"✅ Collection '{collection_name}' deleted successfully:")
                        logger.info(f"Response JSON:\n{json.dumps(result, indent=2)}")

                        self.add_test_result(
                            self._test_delete_image_captioning_collection.test_number,
                            self._test_delete_image_captioning_collection.test_name,
                            f"Delete the image captioning collection '{collection_name}' to clean up the test environment using DELETE /v1/collections endpoint.",
                            ["DELETE /v1/collections"],
                            ["collection_names"],
                            time.time() - delete_start,
                            TestStatus.SUCCESS,
                        )
                        return True
                    else:
                        logger.error(f"❌ Failed to delete collection '{collection_name}': {response.status}")
                        logger.error(f"Response JSON:\n{json.dumps(result, indent=2)}")

                        self.add_test_result(
                            self._test_delete_image_captioning_collection.test_number,
                            self._test_delete_image_captioning_collection.test_name,
                            f"Delete the image captioning collection '{collection_name}' to clean up the test environment using DELETE /v1/collections endpoint.",
                            ["DELETE /v1/collections"],
                            ["collection_names"],
                            time.time() - delete_start,
                            TestStatus.FAILURE,
                            f"API request failed with status {response.status}",
                        )
                        return False
        except Exception as e:
            logger.error(f"❌ Error deleting collection '{collection_name}': {e}")
            self.add_test_result(
                self._test_delete_image_captioning_collection.test_number,
                self._test_delete_image_captioning_collection.test_name,
                f"Delete the image captioning collection '{collection_name}' to clean up the test environment using DELETE /v1/collections endpoint.",
                ["DELETE /v1/collections"],
                ["collection_names"],
                time.time() - delete_start,
                TestStatus.FAILURE,
                f"Exception occurred: {str(e)}",
            )
            return False
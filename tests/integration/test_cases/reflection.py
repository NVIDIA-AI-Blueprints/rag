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
Self Reflection test module

This module contains tests for validating the self-reflection capabilities of the RAG system. It ensures that the system can improve response quality through self-evaluation and regeneration of responses.
"""

import json
import logging
import os
import time

import aiohttp

from ..base import BaseTestModule, TestStatus, test_case
from ..utils.response_handlers import extract_streaming_text, print_response

logger = logging.getLogger(__name__)


class ReflectionModule(BaseTestModule):
    """Reflection test module for validating self-reflection capabilities in RAG responses.

    This module tests the system's ability to improve response quality through
    self-reflection, where the system evaluates and potentially regenerates
    responses to improve accuracy and groundedness.
    """

    COLLECTION_NAME = "test_reflection"
    FILES = [
        "2023 Q3 INTC.pdf",
    ]
    BLOCKING = True

    @test_case(44, "Delete Collection Created for Testing Reflection")
    async def _delete_collection_for_reflection(self) -> bool:
        """Delete collections used for reflection testing.

        This method deletes the specified collection from the system, which is
        used to clean up after reflection tests have been conducted.

        Returns:
            bool: True if collections are deleted successfully, False otherwise
        """
        start_time = time.time()

        async with aiohttp.ClientSession() as session:
            try:
                logger.info("🗑️ Deleting collections:")
                logger.info(
                    f"📋 Collections to delete: {json.dumps(self.COLLECTION_NAME, indent=2)}"
                )

                async with session.delete(
                    f"{self.ingestor_server_url}/v1/collections",
                    params={"collection_names": [self.COLLECTION_NAME]},
                ) as response:
                    result = await response.json()
                    if response.status == 200:
                        logger.info("✅ Collections deleted successfully:")
                        logger.info(f"Response JSON:\n{json.dumps(result, indent=2)}")

                        # Add test result for successful execution
                        self.add_test_result(
                            self._delete_collection_for_reflection.test_number,
                            self._delete_collection_for_reflection.test_name,
                            f"Delete collection '{self.COLLECTION_NAME}' created for reflection testing. Cleans up test environment after reflection tests.",
                            ["DELETE /v1/collections"],
                            ["collection_names"],
                            time.time() - start_time,
                            TestStatus.SUCCESS,
                        )
                        return True
                    else:
                        logger.error(
                            f"❌ Failed to delete collections: {response.status}"
                        )
                        logger.error(f"Response JSON:\n{json.dumps(result, indent=2)}")

                        # Add test result for failed execution
                        self.add_test_result(
                            self._delete_collection_for_reflection.test_number,
                            self._delete_collection_for_reflection.test_name,
                            f"Delete collection '{self.COLLECTION_NAME}' created for reflection testing. Cleans up test environment after reflection tests.",
                            ["DELETE /v1/collections"],
                            ["collection_names"],
                            time.time() - start_time,
                            TestStatus.FAILURE,
                            f"Failed to delete collections: {response.status}",
                        )
                        return False
            except Exception as e:
                logger.error(f"❌ Error deleting collections: {e}")

                # Add test result for exception
                self.add_test_result(
                    self._delete_collection_for_reflection.test_number,
                    self._delete_collection_for_reflection.test_name,
                    f"Delete collection '{self.COLLECTION_NAME}' created for reflection testing. Cleans up test environment after reflection tests.",
                    ["DELETE /v1/collections"],
                    ["collection_names"],
                    time.time() - start_time,
                    TestStatus.FAILURE,
                    f"Error deleting collections: {e}",
                )
                return False

    @test_case(42, "Upload Documents for Testing Reflection")
    async def _upload_documents_for_reflection(self):
        """Upload documents to a collection for reflection testing.

        This method uploads a set of predefined documents to a specified collection
        to facilitate testing of the reflection capabilities of the system.
        """
        start_time = time.time()

        data = {
            "collection_name": self.COLLECTION_NAME,
            "split_options": {"chunk_size": 512, "chunk_overlap": 150},
            "blocking": self.BLOCKING,
        }

        form_data = aiohttp.FormData()
        for file in self.FILES:
            file_path = "./tests/data/" + file
            if os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    file_content = f.read()
                form_data.add_field(
                    "documents",
                    file_content,
                    filename=os.path.basename(file_path),
                    content_type="application/octet-stream",
                )

        form_data.add_field("data", json.dumps(data), content_type="application/json")

        async with aiohttp.ClientSession() as session:
            try:
                logger.info(
                    f"📤 Uploading {len(self.FILES)} documents to collection '{self.COLLECTION_NAME}'"
                )
                logger.info(f"📁 Files: {self.FILES}")
                logger.info(f"📋 Upload data: {json.dumps(data, indent=2)}")

                async with session.post(
                    f"{self.ingestor_server_url}/v1/documents", data=form_data
                ) as response:
                    result = await response.json()
                    if response.status == 200:
                        logger.info(
                            f"✅ Upload request successful. Response:\n{json.dumps(result, indent=2)}"
                        )
                        if self.BLOCKING:
                            # For blocking uploads, the API returns completion result directly
                            total_documents = result.get("total_documents", 0)
                            failed_documents = result.get("failed_documents", [])
                            logger.info(
                                f"✅ Documents uploaded successfully (blocking). Total: {total_documents}, Failed: {len(failed_documents)}"
                            )
                            if failed_documents:
                                logger.warning(
                                    f"⚠️ Failed documents: {failed_documents}"
                                )

                            # Add test result for successful blocking upload
                            self.add_test_result(
                                self._upload_documents_for_reflection.test_number,
                                self._upload_documents_for_reflection.test_name,
                                f"Upload {len(self.FILES)} documents to collection '{self.COLLECTION_NAME}' with chunk size 512 and overlap 150 for reflection testing",
                                ["POST /v1/documents"],
                                ["collection_name", "split_options", "blocking"],
                                time.time() - start_time,
                                TestStatus.SUCCESS,
                            )
                            # Return a special value to indicate blocking completion
                            return "BLOCKING_COMPLETED"
                        else:
                            # For non-blocking uploads, return the task_id
                            task_id = result.get("task_id")
                            logger.info(
                                f"✅ Documents uploaded successfully. Task ID: {task_id}"
                            )

                            # Add test result for successful non-blocking upload
                            self.add_test_result(
                                self._upload_documents_for_reflection.test_number,
                                self._upload_documents_for_reflection.test_name,
                                f"Upload {len(self.FILES)} documents to collection '{self.COLLECTION_NAME}' with chunk size 512 and overlap 150 for reflection testing",
                                ["POST /v1/documents"],
                                ["collection_name", "split_options", "blocking"],
                                time.time() - start_time,
                                TestStatus.SUCCESS,
                            )
                            return task_id
                    else:
                        logger.error(
                            f"❌ Failed to upload documents. Status: {response.status}"
                        )
                        logger.error(f"❌ Response:\n{json.dumps(result, indent=2)}")

                        # Add test result for failed upload
                        self.add_test_result(
                            self._upload_documents_for_reflection.test_number,
                            self._upload_documents_for_reflection.test_name,
                            f"Upload {len(self.FILES)} documents to collection '{self.COLLECTION_NAME}' with chunk size 512 and overlap 150 for reflection testing",
                            ["POST /v1/documents"],
                            ["collection_name", "split_options", "blocking"],
                            time.time() - start_time,
                            TestStatus.FAILURE,
                            f"Failed to upload documents: {response.status}",
                        )
                        return None
            except Exception as e:
                logger.error(f"❌ Error uploading documents: {e}")

                # Add test result for exception
                self.add_test_result(
                    self._upload_documents_for_reflection.test_number,
                    self._upload_documents_for_reflection.test_name,
                    f"Upload {len(self.FILES)} documents to collection '{self.COLLECTION_NAME}' with chunk size 512 and overlap 150 for reflection testing",
                    ["POST /v1/documents"],
                    ["collection_name", "split_options", "blocking"],
                    time.time() - start_time,
                    TestStatus.FAILURE,
                    f"Error uploading documents: {e}",
                )
                return None

    @test_case(41, "Create Collection for Testing Reflection")
    async def _create_collection_for_reflection(self):
        """Create a collection with optional metadata schema for reflection testing.

        This method creates a new collection in the system, which is used to store
        documents for testing the reflection capabilities of the RAG system.
        """
        start_time = time.time()

        try:
            payload = {
                "collection_name": self.COLLECTION_NAME,
                "embedding_dimension": 2048,
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ingestor_server_url}/v1/collection", json=payload
                ) as response:
                    result = await response.json()
                    if response.status == 200:
                        logger.info(
                            f"✅ Collection '{self.COLLECTION_NAME}' created successfully:"
                        )
                        logger.info(f"Response JSON:\n{json.dumps(result, indent=2)}")

                        # Add test result for successful execution
                        self.add_test_result(
                            self._create_collection_for_reflection.test_number,
                            self._create_collection_for_reflection.test_name,
                            f"Create collection '{self.COLLECTION_NAME}' with embedding dimension 2048 for reflection testing",
                            ["POST /v1/collection"],
                            ["collection_name", "embedding_dimension"],
                            time.time() - start_time,
                            TestStatus.SUCCESS,
                        )
                        return True
                    else:
                        logger.error(
                            f"❌ Failed to create collection '{self.COLLECTION_NAME}': {response.status}"
                        )
                        logger.error(f"Response JSON:\n{json.dumps(result, indent=2)}")

                        # Add test result for failed execution
                        self.add_test_result(
                            self._create_collection_for_reflection.test_number,
                            self._create_collection_for_reflection.test_name,
                            f"Create collection '{self.COLLECTION_NAME}' with embedding dimension 2048 for reflection testing",
                            ["POST /v1/collection"],
                            ["collection_name", "embedding_dimension"],
                            time.time() - start_time,
                            TestStatus.FAILURE,
                            f"Failed to create collection: {response.status}",
                        )
                        return False
        except Exception as e:
            logger.error(f"❌ Error creating collection '{self.COLLECTION_NAME}': {e}")

            # Add test result for exception
            self.add_test_result(
                self._create_collection_for_reflection.test_number,
                self._create_collection_for_reflection.test_name,
                f"Create collection '{self.COLLECTION_NAME}' with embedding dimension 2048 for reflection testing",
                ["POST /v1/collection"],
                ["collection_name", "embedding_dimension"],
                time.time() - start_time,
                TestStatus.FAILURE,
                f"Error creating collection: {e}",
            )
            return False

    async def _get_response_with_self_reflection_enabled(self) -> str | None:
        """Get response from RAG server with self-reflection enabled.

        This method sends a request to the RAG server with reflection enabled
        to test if the system can improve response quality through self-evaluation.

        Returns:
            str | None: The response text from the server, or None if request failed
        """
        logger.info("Starting request to RAG server with self-reflection enabled")
        # Construct payload with reflection-specific parameters
        # Using a test question about Intel's effective tax rate to evaluate
        # the system's ability to analyze and improve its response quality
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": "Examine how Intel's effective tax rate in the most recent 10-Q compares with the tax-related discussions in the notes section",
                }
            ],
            "collection_names": [self.COLLECTION_NAME],
            "enable_citations": True,  # Enable citations for better groundedness
            "reranker_top_k": 5,  # Limit reranked results for focused analysis
            "vdb_top_k": 10,  # Vector DB top-k for initial retrieval
            "enable_reranker": True,  # Enable reranking for better relevance
        }

        logger.debug(f"Request payload prepared: {payload}")

        # Make HTTP request to RAG server with reflection-enabled payload
        logger.info(f"Sending POST request to {self.rag_server_url}/v1/generate")
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.rag_server_url}/v1/generate", json=payload
            ) as response:
                logger.info(f"Received response with status: {response.status}")
                result = await print_response(response)
                if response.status == 200:
                    logger.info("Successfully received response from server")
                    if result.get("streaming_response"):
                        # Extract text from streaming response format
                        logger.debug("Processing streaming response format")
                        response_text = extract_streaming_text(result)
                        logger.debug(
                            f"Extracted streaming text length: {len(response_text) if response_text else 0}"
                        )
                    else:
                        # Extract text from standard JSON response format
                        logger.debug("Processing standard JSON response format")
                        choices = result.get("choices", [])
                        if choices:
                            response_text = (
                                choices[0].get("message", {}).get("content", "")
                            )
                            logger.debug(
                                f"Extracted response text length: {len(response_text)}"
                            )
                        else:
                            logger.warning("No choices found in response")
                            response_text = ""
                    return response_text
                else:
                    logger.error(f"Request failed with status {response.status}")
                    return None

        # Fallback return if no session was created or other issues
        logger.error("Failed to establish session or get response")
        return None

    @test_case(43, "Test Self Reflection")
    async def _test_self_reflection(self) -> bool:
        """Test the self-reflection capability of the RAG system.

        This test validates that when reflection is enabled, the system can:
        1. Generate an initial response
        2. Evaluate the response quality
        3. Regenerate a better response if needed

        The test measures response time and validates that a response is re-generated,
        indicating the reflection mechanism is functioning.

        Returns:
            bool: True if reflection test passes, False otherwise
        """
        logger.info("Starting self-reflection test case")
        try:
            # Start timing the reflection process to measure performance impact
            logger.info("Beginning reflection response generation timing")
            reflection_start = time.time()

            # Get response with reflection enabled - this may take longer than
            # standard responses due to the self-evaluation and potential regeneration
            resp = await self._get_response_with_self_reflection_enabled()

            # Calculate total time taken for reflection-enabled response
            reflection_time = time.time() - reflection_start
            logger.info(
                f"Reflection process completed in {reflection_time:.2f} seconds"
            )

            # Validate that a response was re-generated successfully
            if resp and ("new, grounded response" in resp.lower() or \
                "new response" in resp.lower() or "response" in resp.lower()):
                logger.info(
                    "Self-reflection test succeeded - response re-generated successfully"
                )
                logger.info(f"Response preview: {resp}")

                # Record successful test result with reflection metrics
                self.add_test_result(
                    self._test_self_reflection.test_number,
                    self._test_self_reflection.test_name,
                    "Verify that due to self reflection, the response has been re-generated.",
                    ["POST /v1/generate"],  # Corrected HTTP method
                    [
                        "enable_reflection",
                    ],
                    reflection_time,
                    TestStatus.SUCCESS,
                )
                return True
            else:
                # Handle case where response was not re-generated
                logger.error(
                    "Self-reflection test failed - response is not re-generated"
                )

                # Record failed test result with detailed error information
                self.add_test_result(
                    self._test_self_reflection.test_number,
                    self._test_self_reflection.test_name,
                    "Verify that due to self reflection, the response has been re-generated.",
                    ["POST /v1/generate"],  # Corrected HTTP method
                    [
                        "enable_reflection",
                    ],
                    reflection_time,
                    TestStatus.FAILURE,
                    "Failed to re-generate a response with self reflection enabled",
                )
                return False
        except Exception as e:
            # Handle any unexpected errors during the reflection test
            logger.error(f"Self-reflection test encountered an exception: {str(e)}")
            logger.exception(
                "Full exception details:"
            )  # This will log the full traceback

            # Ensure reflection_time is defined even if error occurred early
            if "reflection_time" not in locals():
                reflection_time = 0.0
                logger.warning("Setting reflection_time to 0.0 due to early exception")

            # Record failed test result with exception details
            self.add_test_result(
                self._test_self_reflection.test_number,
                self._test_self_reflection.test_name,
                "Verify that due to self reflection, the response has been re-generated.",
                ["POST /v1/generate"],  # Corrected HTTP method
                [
                    "enable_reflection",
                ],
                reflection_time,
                TestStatus.FAILURE,
                f"Self reflection test failed due to error: {e}",
            )
            return False

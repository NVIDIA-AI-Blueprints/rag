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
Output sequence length validation module

This module contains integration tests to validate that the RAG server honors
`min_tokens` and `max_tokens` generation parameters. It uploads a small set of
documents, calls the `/v1/generate` API with explicit token limits, and
verifies the number of output tokens using `tiktoken`.
"""

import json
import logging
import os
import time

import aiohttp
import tiktoken

from ..base import BaseTestModule, TestStatus, test_case
from ..utils.response_handlers import extract_streaming_text, print_response

logger = logging.getLogger(__name__)


class OutputSequenceLengthValidationModule(BaseTestModule):
    """Validate that `min_tokens` and `max_tokens` are enforced by /v1/generate.

    The tests upload a small document to a dedicated collection and then issue
    generation requests with specific `min_tokens` and `max_tokens` values. The
    output token count is computed locally using `tiktoken` and compared against
    the requested limits.
    """

    COLLECTION_NAME = "test_osl"
    FILES = [
        "2023 Q3 INTC.pdf",
    ]
    BLOCKING = True

    # Test parameters
    MIN_TOKENS = 128  # also used as max_tokens to enforce OSL

    @test_case(70, "Delete Collection Created for Output Sequence Length Validation")
    async def _delete_collection_for_osl_validation(self) -> bool:
        """Delete collection used for output sequence length validation.

        Returns:
            bool: True if collection is deleted successfully, False otherwise
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
                    json=[self.COLLECTION_NAME],
                ) as response:
                    result = await response.json()
                    if response.status == 200:
                        logger.info("✅ Collections deleted successfully:")
                        logger.info(f"Response JSON:\n{json.dumps(result, indent=2)}")

                        # Add test result for successful execution
                        self.add_test_result(
                            self._delete_collection_for_osl_validation.test_number,
                            self._delete_collection_for_osl_validation.test_name,
                            f"Delete collection '{self.COLLECTION_NAME}' used for output sequence length validation.",
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
                            self._delete_collection_for_osl_validation.test_number,
                            self._delete_collection_for_osl_validation.test_name,
                            f"Delete collection '{self.COLLECTION_NAME}' used for output sequence length validation.",
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
                    self._delete_collection_for_osl_validation.test_number,
                    self._delete_collection_for_osl_validation.test_name,
                    f"Delete collection '{self.COLLECTION_NAME}' used for output sequence length validation.",
                    ["DELETE /v1/collections"],
                    ["collection_names"],
                    time.time() - start_time,
                    TestStatus.FAILURE,
                    f"Error deleting collections: {e}",
                )
                return False

    @test_case(68, "Upload Documents for Output Sequence Length Validation")
    async def _upload_documents_for_osl_validation(self):
        """Upload documents to a collection for output sequence length validation."""
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
                                self._upload_documents_for_osl_validation.test_number,
                                self._upload_documents_for_osl_validation.test_name,
                                f"Upload {len(self.FILES)} documents to collection '{self.COLLECTION_NAME}' with chunk size 512 and overlap 150 for output sequence length validation",
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
                                self._upload_documents_for_osl_validation.test_number,
                                self._upload_documents_for_osl_validation.test_name,
                                f"Upload {len(self.FILES)} documents to collection '{self.COLLECTION_NAME}' with chunk size 512 and overlap 150 for output sequence length validation",
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
                            self._upload_documents_for_osl_validation.test_number,
                            self._upload_documents_for_osl_validation.test_name,
                            f"Upload {len(self.FILES)} documents to collection '{self.COLLECTION_NAME}' with chunk size 512 and overlap 150 for output sequence length validation",
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
                    self._upload_documents_for_osl_validation.test_number,
                    self._upload_documents_for_osl_validation.test_name,
                    f"Upload {len(self.FILES)} documents to collection '{self.COLLECTION_NAME}' with chunk size 512 and overlap 150 for output sequence length validation",
                    ["POST /v1/documents"],
                    ["collection_name", "split_options", "blocking"],
                    time.time() - start_time,
                    TestStatus.FAILURE,
                    f"Error uploading documents: {e}",
                )
                return None

    @test_case(67, "Create Collection for Output Sequence Length Validation")
    async def _create_collection_for_osl_validation(self):
        """Create a collection used for output sequence length validation."""
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
                            self._create_collection_for_osl_validation.test_number,
                            self._create_collection_for_osl_validation.test_name,
                            f"Create collection '{self.COLLECTION_NAME}' with embedding dimension 2048 for output sequence length validation",
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
                            self._create_collection_for_osl_validation.test_number,
                            self._create_collection_for_osl_validation.test_name,
                            f"Create collection '{self.COLLECTION_NAME}' with embedding dimension 2048 for output sequence length validation",
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
                self._create_collection_for_osl_validation.test_number,
                self._create_collection_for_osl_validation.test_name,
                f"Create collection '{self.COLLECTION_NAME}' with embedding dimension 2048 for output sequence length validation",
                ["POST /v1/collection"],
                ["collection_name", "embedding_dimension"],
                time.time() - start_time,
                TestStatus.FAILURE,
                f"Error creating collection: {e}",
            )
            return False

    async def _get_generated_text(self, *, min_tokens: int | None, max_tokens: int | None, ignore_eos: bool = True) -> str | None:
        """Call /v1/generate with token parameters and return the generated text."""
        logger.info("Starting request to RAG server with token limits")
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": "Summarize the key financial highlights from the uploaded Intel document.",
                }
            ],
            "collection_names": [self.COLLECTION_NAME],
            "enable_citations": True, 
            "reranker_top_k": 5,
            "vdb_top_k": 10,
            "enable_reranker": True,
        }
        if min_tokens is not None:
            payload["min_tokens"] = min_tokens
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        payload["ignore_eos"] = ignore_eos

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
                        logger.debug("Processing streaming response format")
                        response_text = extract_streaming_text(result)
                        logger.debug(
                            f"Extracted streaming text length: {len(response_text) if response_text else 0}"
                        )
                    else:
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

    def _count_output_tokens(self, text: str) -> int:
        """Count tokens in text using cl100k_base encoding."""
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text or ""))
        except Exception as e:
            logger.error(f"❌ Error counting tokens: {e}")
            return 0

    @test_case(69, "Validate Output Sequence Length")
    async def _validate_output_sequence_length(self) -> bool:
        """Validate output sequence length equals specified tokens when min==max.

        Ensures the number of generated output tokens equals the requested
        `min_tokens` when `min_tokens == max_tokens` and `ignore_eos=True`.
        """
        logger.info("Starting output sequence length validation test")
        start = time.time()
        try:
            resp_text = await self._get_generated_text(
                min_tokens=self.MIN_TOKENS,
                max_tokens=self.MIN_TOKENS,
                ignore_eos=True,
            )
            elapsed = time.time() - start
            if not resp_text:
                self.add_test_result(
                    self._validate_output_sequence_length.test_number,
                    self._validate_output_sequence_length.test_name,
                    "Validate output sequence length for min_tokens==max_tokens",
                    ["POST /v1/generate"],
                    ["min_tokens", "max_tokens", "ignore_eos"],
                    elapsed,
                    TestStatus.FAILURE,
                    "No response text returned",
                )
                return False

            token_count = self._count_output_tokens(resp_text)
            logger.info(f"Got output sequence length as {token_count}")
            expected = self.MIN_TOKENS
            if token_count >= expected-50 and token_count <= expected+50:
                self.add_test_result(
                    self._validate_output_sequence_length.test_number,
                    self._validate_output_sequence_length.test_name,
                    f"Validate output token count {token_count} is within 50 tokens of {expected}",
                    ["POST /v1/generate"],
                    ["min_tokens", "max_tokens", "ignore_eos"],
                    elapsed,
                    TestStatus.SUCCESS,
                )
                return True
            else:
                logger.error(f"❌ Expected {expected} tokens, got {token_count}. Token count is not within 50 tokens of {expected}")
                self.add_test_result(
                    self._validate_output_sequence_length.test_number,
                    self._validate_output_sequence_length.test_name,
                    f"Validate output token count {token_count} is within 50 tokens of {expected}",
                    ["POST /v1/generate"],
                    ["min_tokens", "max_tokens", "ignore_eos"],
                    elapsed,
                    TestStatus.FAILURE,
                    f"Expected {expected} tokens, got {token_count}",
                )
                return False
        except Exception as e:
            elapsed = time.time() - start
            self.add_test_result(
                self._validate_output_sequence_length.test_number,
                self._validate_output_sequence_length.test_name,
                "Validate output sequence length for min_tokens==max_tokens",
                ["POST /v1/generate"],
                ["min_tokens", "max_tokens", "ignore_eos"],
                elapsed,
                TestStatus.FAILURE,
                f"Error validating output sequence length: {e}",
            )
            return False

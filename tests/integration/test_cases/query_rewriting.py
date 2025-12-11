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
Query rewriting test module
"""

import json
import logging
import os
import time

import aiohttp

from ..base import BaseTestModule, TestStatus, test_case
from ..utils.response_handlers import extract_streaming_text, print_response

logger = logging.getLogger(__name__)


class QueryRewritingModule(BaseTestModule):
    """Integration test module for validating query rewriting capabilities.

    This module tests the system's ability to rewrite queries to improve retrieval
    accuracy and relevance. It includes tests for creating collections, uploading
    documents, and validating query rewriting in both search and generate APIs.
    """

    COLLECTION_NAME = "test_query_rewriting"
    FILES = [
        "query_rewriting.pdf",
    ]
    BLOCKING = True

    @test_case(47, "Delete Collection Created for Testing Query Rewriting")
    async def _delete_collection_for_query_rewriting(self) -> bool:
        """Delete collections used for query rewriting testing.

        This method deletes the specified collection from the system, which is
        used to clean up after query rewriting tests have been conducted.

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
                            self._delete_collection_for_query_rewriting.test_number,
                            self._delete_collection_for_query_rewriting.test_name,
                            f"Delete collection '{self.COLLECTION_NAME}' created for query rewriting testing. Cleans up test environment after query rewriting tests.",
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
                            self._delete_collection_for_query_rewriting.test_number,
                            self._delete_collection_for_query_rewriting.test_name,
                            f"Delete collection '{self.COLLECTION_NAME}' created for query rewriting testing. Cleans up test environment after query rewriting tests.",
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
                    self._delete_collection_for_query_rewriting.test_number,
                    self._delete_collection_for_query_rewriting.test_name,
                    f"Delete collection '{self.COLLECTION_NAME}' created for query rewriting testing. Cleans up test environment after query rewriting tests.",
                    ["DELETE /v1/collections"],
                    ["collection_names"],
                    time.time() - start_time,
                    TestStatus.FAILURE,
                    f"Error deleting collections: {e}",
                )
                return False

    @test_case(46, "Upload Documents for Testing Query Rewriting")
    async def _upload_documents_for_query_rewriting(self):
        """Upload documents to a collection for query rewriting testing.

        This method uploads a set of predefined documents to a specified collection
        to facilitate testing of the query rewriting capabilities of the system.
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
                                self._upload_documents_for_query_rewriting.test_number,
                                self._upload_documents_for_query_rewriting.test_name,
                                f"Upload {len(self.FILES)} documents to collection '{self.COLLECTION_NAME}' with chunk size 512 and overlap 150 for query rewriting testing",
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
                                self._upload_documents_for_query_rewriting.test_number,
                                self._upload_documents_for_query_rewriting.test_name,
                                f"Upload {len(self.FILES)} documents to collection '{self.COLLECTION_NAME}' with chunk size 512 and overlap 150 for query rewriting testing",
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
                            self._upload_documents_for_query_rewriting.test_number,
                            self._upload_documents_for_query_rewriting.test_name,
                            f"Upload {len(self.FILES)} documents to collection '{self.COLLECTION_NAME}' with chunk size 512 and overlap 150 for query rewriting testing",
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
                    self._upload_documents_for_query_rewriting.test_number,
                    self._upload_documents_for_query_rewriting.test_name,
                    f"Upload {len(self.FILES)} documents to collection '{self.COLLECTION_NAME}' with chunk size 512 and overlap 150 for query rewriting testing",
                    ["POST /v1/documents"],
                    ["collection_name", "split_options", "blocking"],
                    time.time() - start_time,
                    TestStatus.FAILURE,
                    f"Error uploading documents: {e}",
                )
                return None

    @test_case(45, "Create Collection for Testing Query Rewriting")
    async def _create_collection_for_query_rewriting(self):
        """Create a collection with optional metadata schema for query rewriting testing.

        This method creates a new collection in the system, which is used to store
        documents for testing the query rewriting capabilities of the RAG system.
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
                            self._create_collection_for_query_rewriting.test_number,
                            self._create_collection_for_query_rewriting.test_name,
                            f"Create collection '{self.COLLECTION_NAME}' with embedding dimension 2048 for query rewriting testing",
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
                            self._create_collection_for_query_rewriting.test_number,
                            self._create_collection_for_query_rewriting.test_name,
                            f"Create collection '{self.COLLECTION_NAME}' with embedding dimension 2048 for query rewriting testing",
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
                self._create_collection_for_query_rewriting.test_number,
                self._create_collection_for_query_rewriting.test_name,
                f"Create collection '{self.COLLECTION_NAME}' with embedding dimension 2048 for query rewriting testing",
                ["POST /v1/collection"],
                ["collection_name", "embedding_dimension"],
                time.time() - start_time,
                TestStatus.FAILURE,
                f"Error creating collection: {e}",
            )
            return False

    async def _call_query_rewriting_search(self) -> str | None:
        """Call the search API with query rewriting enabled and validate the results.

        This method sends a search request to the RAG server with query rewriting
        enabled and checks if the correct document and page number are retrieved.

        Returns:
            str | None: The document name and page number if successful, None otherwise
        """
        try:
            payload = {
                "query": "When was the prime minister of India born?",
                "reranker_top_k": 1,
                "vdb_top_k": 10,
                "collection_names": [self.COLLECTION_NAME],
                "messages": [
                    {
                        "role": "user",
                        "content": "Who is the prime minister of India?",
                    },
                    {
                        "role": "assistant",
                        "content": "Narendra Modi is the prime minister of India",
                    },
                    {
                        "role": "user",
                        "content": "When was he born?",
                    },
                ],
                "enable_query_rewriting": True,
                "enable_reranker": True,
            }

            async with aiohttp.ClientSession() as session:
                try:
                    logger.info("🔍 Searching with citations")
                    logger.info(
                        f"📋 Search request payload:\n{json.dumps(payload, indent=2)}"
                    )

                    async with session.post(
                        f"{self.rag_server_url}/v1/search", json=payload
                    ) as response:
                        result = await print_response(response)
                        if response.status == 200:
                            # Check if citations are present in the results array
                            results = result.get("results", [])
                            if results:
                                for doc in results:
                                    if doc.get("document_name", None):
                                        return (
                                            doc.get("document_name")
                                            + "_"
                                            + str(
                                                doc.get("metadata", {})
                                                .get("content_metadata", {})
                                                .get("page_number")
                                            )
                                        )
                            else:
                                logger.error(
                                    "⚠️ Search test passed but no results found"
                                )
                                return None
                        else:
                            logger.error("❌ Search with citations test failed")
                            return None
                except Exception as e:
                    logger.error(f"❌ Error in search with citations test: {e}")
                    return None
        except Exception as e:
            logger.error(f"Query rewriting test encountered an exception: {e}")
            return None

    async def _call_query_rewriting_generate(self) -> str | None:
        """Call the generate API with query rewriting enabled and validate the results.

        This method sends a generate request to the RAG server with query rewriting
        enabled and checks if the response contains the expected content.

        Returns:
            str | None: The response text if successful, None otherwise
        """
        try:
            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": "Who is the prime minister of India?",
                    },
                    {
                        "role": "assistant",
                        "content": "Narendra Modi is the prime minister of India",
                    },
                    {
                        "role": "user",
                        "content": "When was he born?",
                    },
                ],
                "collection_names": [self.COLLECTION_NAME],
                "enable_citations": True,
                "reranker_top_k": 1,
                "vdb_top_k": 10,
                "enable_query_rewriting": True,
                "enable_reranker": True,
            }
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
                        logger.info(f"Response preview: {response_text}")
                        return response_text
                    else:
                        logger.error(f"Request failed with status {response.status}")
                        return None
        except Exception as e:
            logger.error(f"Query rewriting test encountered an exception: {e}")
            return None

    @test_case(48, "Test Query Rewriting in Search API")
    async def _test_query_rewriting_search(self) -> bool:
        """Test the query rewriting capability in the search API.

        This test validates that when query rewriting is enabled, the system can
        fetch the correct document and page number from the ingested documents.

        Returns:
            bool: True if the search test passes, False otherwise
        """
        try:
            query_rewriting_start = time.time()

            res = await self._call_query_rewriting_search()
            query_rewriting_time = time.time() - query_rewriting_start
            logger.info(
                f"Query rewriting process completed in {query_rewriting_time:.2f} seconds with results: {res}"
            )

            if res and res == "query_rewriting.pdf_2":
                self.add_test_result(
                    self._test_query_rewriting_search.test_number,
                    self._test_query_rewriting_search.test_name,
                    "Verify that query rewriting correctly retrieves the expected document filename and page number by leveraging chat history context from the ingested document collection",
                    ["POST /v1/search"],  # Corrected HTTP method
                    [
                        "enable_query_rewriting",
                    ],
                    query_rewriting_time,
                    TestStatus.SUCCESS,
                )
                return True
            else:
                # Record failed test result with detailed error information
                self.add_test_result(
                    self._test_query_rewriting_search.test_number,
                    self._test_query_rewriting_search.test_name,
                    "Verify that query rewriting correctly retrieves the expected document filename and page number by leveraging chat history context from the ingested document collection",
                    ["POST /v1/search"],  # Corrected HTTP method
                    [
                        "enable_query_rewriting",
                    ],
                    query_rewriting_time,
                    TestStatus.FAILURE,
                    "Query rewriting test failed to retrieve the expected document filename and page number when leveraging chat history context from the ingested document collection",
                )
                return False
        except Exception as e:
            logger.error(f"Query rewriting test encountered an exception: {e}")
            logger.exception("Full exception details:")

            if "query_rewriting_time" not in locals():
                query_rewriting_time = 0.0
                logger.warning(
                    "Setting query_rewriting_time to 0.0 due to early exception"
                )

            # Record failed test result with exception details
            self.add_test_result(
                self._test_query_rewriting_search.test_number,
                self._test_query_rewriting_search.test_name,
                "Verify that query rewriting correctly retrieves the expected document filename and page number by leveraging chat history context from the ingested document collection",
                ["POST /v1/search"],  # Corrected HTTP method
                [
                    "enable_query_rewriting",
                ],
                query_rewriting_time,
                TestStatus.FAILURE,
                f"Query rewriting test failed due to error: {e}",
            )
            return False

    @test_case(49, "Test Query Rewriting in Generate API")
    async def _test_query_rewriting_generate(self) -> bool:
        """Test the query rewriting capability in the generate API.

        This test validates that when query rewriting is enabled, the system can
        generate responses that contain the expected content.

        Returns:
            bool: True if the generate test passes, False otherwise
        """
        try:
            query_rewriting_start = time.time()
            res = await self._call_query_rewriting_generate()
            query_rewriting_time = time.time() - query_rewriting_start
            logger.info(
                f"Query rewriting process completed in {query_rewriting_time:.2f} seconds"
            )

            if res and "17 September 1950".lower() in res.lower():
                logger.info(
                    "Query rewriting test succeeded - query rewritten successfully"
                )

                self.add_test_result(
                    self._test_query_rewriting_generate.test_number,
                    self._test_query_rewriting_generate.test_name,
                    "Verify that query rewriting generates accurate responses by leveraging chat history context from the ingested document collection",
                    ["POST /v1/generate"],  # Corrected HTTP method
                    [
                        "enable_query_rewriting",
                    ],
                    query_rewriting_time,
                    TestStatus.SUCCESS,
                )
                return True
            else:
                logger.error("Query rewriting test failed")

                self.add_test_result(
                    self._test_query_rewriting_generate.test_number,
                    self._test_query_rewriting_generate.test_name,
                    "Verify that query rewriting generates accurate responses by leveraging chat history context from the ingested document collection",
                    ["POST /v1/generate"],
                    [
                        "enable_query_rewriting",
                    ],
                    query_rewriting_time,
                    TestStatus.FAILURE,
                    "Query rewriting test failed to generate accurate responses when leveraging chat history context from the ingested document collection",
                )
                return False
        except Exception as e:
            logger.error(f"Query rewriting test encountered an exception: {e}")
            logger.exception("Full exception details:")

            if "query_rewriting_time" not in locals():
                query_rewriting_time = 0.0
                logger.warning(
                    "Setting query_rewriting_time to 0.0 due to early exception"
                )
            self.add_test_result(
                self._test_query_rewriting_generate.test_number,
                self._test_query_rewriting_generate.test_name,
                "Verify that query rewriting generates accurate responses by leveraging chat history context from the ingested document collection",
                ["POST /v1/generate"],
                [
                    "enable_query_rewriting",
                ],
                query_rewriting_time,
                TestStatus.FAILURE,
                f"Query rewriting test failed due to error: {e}",
            )
            return False

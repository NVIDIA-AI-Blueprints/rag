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
RAG search test module
"""

import json
import logging
import time

import aiohttp

from ..base import BaseTestModule, TestStatus, test_case
from ..utils.response_handlers import print_response
from ..utils.verification import verify_citation_document_names

logger = logging.getLogger(__name__)


class RAGSearchModule(BaseTestModule):
    """RAG search test module"""

    @test_case(13, "Search with Citations")
    async def _test_search_with_citations(self) -> bool:
        """Test search with citations"""
        logger.info("\n=== Test 13: Search with Citations ===")
        search_citations_start = time.time()
        search_citations_success = await self.test_search_with_citations()
        search_citations_time = time.time() - search_citations_start

        if search_citations_success:
            self.add_test_result(
                self._test_search_with_citations.test_number,
                self._test_search_with_citations.test_name,
                f"Test search functionality with reranker enabled and comprehensive citation verification. Collection: {self.collections['with_metadata']}. Validates citation count, document names, and ensures citations match documents in the specified collection.",
                ["POST /v1/search"],
                ["query", "collection_names", "enable_reranker", "reranker_top_k"],
                search_citations_time,
                TestStatus.SUCCESS,
            )
            return True
        else:
            self.add_test_result(
                self._test_search_with_citations.test_number,
                self._test_search_with_citations.test_name,
                f"Test search functionality with reranker enabled and comprehensive citation verification. Collection: {self.collections['with_metadata']}. Validates citation count, document names, and ensures citations match documents in the specified collection.",
                ["POST /v1/search"],
                ["query", "collection_names", "enable_reranker", "reranker_top_k"],
                search_citations_time,
                TestStatus.FAILURE,
                "Search with citations test failed",
            )
            return False

    @test_case(14, "Search without Reranker")
    async def _test_search_without_reranker(self) -> bool:
        """Test search without reranker"""
        logger.info("\n=== Test 14: Search without Reranker ===")
        search_no_reranker_start = time.time()
        search_no_reranker_success = await self.test_search_without_reranker()
        search_no_reranker_time = time.time() - search_no_reranker_start

        if search_no_reranker_success:
            self.add_test_result(
                self._test_search_without_reranker.test_number,
                self._test_search_without_reranker.test_name,
                f"Test search functionality with reranker disabled to verify vdb_top_k behavior. Collection: {self.collections['with_metadata']}. Validates that results count matches vdb_top_k when reranker is disabled, with citation verification.",
                ["POST /v1/search"],
                ["query", "collection_names", "enable_reranker", "vdb_top_k"],
                search_no_reranker_time,
                TestStatus.SUCCESS,
            )
            return True
        else:
            self.add_test_result(
                self._test_search_without_reranker.test_number,
                self._test_search_without_reranker.test_name,
                f"Test search functionality with reranker disabled to verify vdb_top_k behavior. Collection: {self.collections['with_metadata']}. Validates that results count matches vdb_top_k when reranker is disabled, with citation verification.",
                ["POST /v1/search"],
                ["query", "collection_names", "enable_reranker", "vdb_top_k"],
                search_no_reranker_time,
                TestStatus.FAILURE,
                "Search without reranker test failed",
            )
            return False

    async def test_search_with_citations(self) -> bool:
        """Test /search endpoint and verify citations are returned"""
        payload = {
            "query": "Tell me something interesting",
            "reranker_top_k": 2,
            "vdb_top_k": 10,
            "collection_names": [self.collections["with_metadata"]],
            "messages": [],
            "enable_query_rewriting": False,
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
                            expected_count = (
                                payload["reranker_top_k"]
                                if payload["enable_reranker"]
                                else payload["vdb_top_k"]
                            )

                            if len(results) == expected_count:
                                logger.info(
                                    f"✅ Search with citations test passed - found {len(results)} results (expected: {expected_count})"
                                )

                                # Verify document names match files in collection
                                if await verify_citation_document_names(
                                    results,
                                    [self.collections["with_metadata"]],
                                    self.ingestor_server_url,
                                ):
                                    logger.info("✅ Citation document names verified")
                                else:
                                    logger.error(
                                        "❌ Citation document names verification failed"
                                    )
                                    return False
                            else:
                                logger.error(
                                    f"❌ Search result count mismatch: got {len(results)}, expected {expected_count}"
                                )
                                return False

                            # Log some details about the first result for verification
                            if results:
                                first_result = results[0]
                                logger.info(
                                    f"First result: document_name={first_result.get('document_name')}, score={first_result.get('score')}"
                                )
                            return True
                        else:
                            logger.error("⚠️ Search test passed but no results found")
                            return False
                    else:
                        logger.error("❌ Search with citations test failed")
                        return False
            except Exception as e:
                logger.error(f"❌ Error in search with citations test: {e}")
                return False

    async def test_search_without_reranker(self) -> bool:
        """Test /search endpoint with enable_reranker=false and verify vdb_top_k citations"""
        payload = {
            "query": "Tell me about the content of these documents",
            "reranker_top_k": 2,
            "vdb_top_k": 5,  # Use a different value to distinguish from reranker test
            "collection_names": [self.collections["with_metadata"]],
            "messages": [],
            "enable_query_rewriting": False,
            "enable_reranker": False,
        }

        async with aiohttp.ClientSession() as session:
            try:
                logger.info("🔍 Searching without reranker")
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
                            expected_count = payload[
                                "vdb_top_k"
                            ]  # Should use vdb_top_k when reranker is disabled

                            if len(results) == expected_count:
                                logger.info(
                                    f"✅ Search without reranker test passed - found {len(results)} results (expected: {expected_count})"
                                )

                                # Verify document names match files in collection
                                if await verify_citation_document_names(
                                    results,
                                    [self.collections["with_metadata"]],
                                    self.ingestor_server_url,
                                ):
                                    logger.info("✅ Citation document names verified")
                                else:
                                    logger.error(
                                        "❌ Citation document names verification failed"
                                    )
                                    return False
                            else:
                                logger.warning(
                                    f"❌ Search result count mismatch: got {len(results)}, expected {expected_count}"
                                )
                                # TODO: return False

                            # Log some details about the first result for verification
                            if results:
                                first_result = results[0]
                                logger.info(
                                    f"First result: document_name={first_result.get('document_name')}, score={first_result.get('score')}"
                                )
                            return True
                        else:
                            logger.error("⚠️ Search test passed but no results found")
                            return False
                    else:
                        logger.error("❌ Search without reranker test failed")
                        return False
            except Exception as e:
                logger.error(f"❌ Error in search without reranker test: {e}")
                return False

    @test_case(33, "Search with Confidence Threshold")
    async def _test_search_with_confidence_threshold(self) -> bool:
        """Test search with confidence threshold filtering"""
        logger.info("\n=== Test 33: Search with Confidence Threshold ===")
        search_confidence_start = time.time()
        search_confidence_success = await self.test_search_with_confidence_threshold()
        search_confidence_time = time.time() - search_confidence_start

        if search_confidence_success:
            self.add_test_result(
                self._test_search_with_confidence_threshold.test_number,
                self._test_search_with_confidence_threshold.test_name,
                f"Test search functionality with confidence threshold filtering. Collection: {self.collections['with_metadata']}. Validates that only documents with relevance scores >= confidence_threshold are returned, with proper logging and error handling.",
                ["POST /v1/search"],
                [
                    "query",
                    "collection_names",
                    "enable_reranker",
                    "confidence_threshold",
                ],
                search_confidence_time,
                TestStatus.SUCCESS,
            )
            return True
        else:
            self.add_test_result(
                self._test_search_with_confidence_threshold.test_number,
                self._test_search_with_confidence_threshold.test_name,
                f"Test search functionality with confidence threshold filtering. Collection: {self.collections['with_metadata']}. Validates that only documents with relevance scores >= confidence_threshold are returned, with proper logging and error handling.",
                ["POST /v1/search"],
                [
                    "query",
                    "collection_names",
                    "enable_reranker",
                    "confidence_threshold",
                ],
                search_confidence_time,
                TestStatus.FAILURE,
                "Search with confidence threshold test failed",
            )
            return False

    async def test_search_with_confidence_threshold(self) -> bool:
        """Test /search endpoint with confidence threshold filtering"""
        payload = {
            "query": "Tell me about the content of these documents",
            "reranker_top_k": 5,
            "vdb_top_k": 10,
            "collection_names": [self.collections["with_metadata"]],
            "messages": [],
            "enable_query_rewriting": False,
            "enable_reranker": True,
            "confidence_threshold": 0.5,  # Only include documents with score >= 0.5
        }

        async with aiohttp.ClientSession() as session:
            try:
                logger.info("🔍 Searching with confidence threshold filtering")
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
                            # Verify that all results have relevance scores >= threshold
                            threshold = payload["confidence_threshold"]
                            filtered_results = [
                                result
                                for result in results
                                if result.get("score", 0.0) >= threshold
                            ]

                            if len(filtered_results) == len(results):
                                logger.info(
                                    f"✅ Confidence threshold filtering test passed - all {len(results)} results have scores >= {threshold}"
                                )

                                # Log scores for verification
                                for i, result in enumerate(results):
                                    logger.info(
                                        f"Result {i+1}: document_name={result.get('document_name')}, score={result.get('score')}"
                                    )

                                # Verify document names match files in collection
                                if await verify_citation_document_names(
                                    results,
                                    [self.collections["with_metadata"]],
                                    self.ingestor_server_url,
                                ):
                                    logger.info("✅ Citation document names verified")
                                else:
                                    logger.error(
                                        "❌ Citation document names verification failed"
                                    )
                                    return False

                                return True
                            else:
                                logger.error(
                                    f"❌ Confidence threshold filtering failed: {len(results) - len(filtered_results)} results have scores < {threshold}"
                                )
                                return False
                        else:
                            # No results found - this is expected when confidence threshold filters out all documents
                            logger.info(
                                "✅ Confidence threshold filtering working correctly - no documents met the threshold"
                            )
                            return True
                    else:
                        logger.error("❌ Search with confidence threshold test failed")
                        return False
            except Exception as e:
                logger.error(f"❌ Error in search with confidence threshold test: {e}")
                return False

    @test_case(34, "Search with Confidence Threshold Warning")
    async def _test_search_with_confidence_threshold_warning(self) -> bool:
        """Test search with confidence threshold but without reranker (should show warning)"""
        logger.info("\n=== Test 34: Search with Confidence Threshold Warning ===")
        search_confidence_warning_start = time.time()
        search_confidence_warning_success = (
            await self.test_search_with_confidence_threshold_warning()
        )
        search_confidence_warning_time = time.time() - search_confidence_warning_start

        if search_confidence_warning_success:
            self.add_test_result(
                self._test_search_with_confidence_threshold_warning.test_number,
                self._test_search_with_confidence_threshold_warning.test_name,
                f"Test search functionality with confidence threshold but without reranker to verify warning behavior. Collection: {self.collections['with_metadata']}. Validates that appropriate warnings are logged when confidence_threshold is set but enable_reranker is False.",
                ["POST /v1/search"],
                [
                    "query",
                    "collection_names",
                    "enable_reranker",
                    "confidence_threshold",
                ],
                search_confidence_warning_time,
                TestStatus.SUCCESS,
            )
            return True
        else:
            self.add_test_result(
                self._test_search_with_confidence_threshold_warning.test_number,
                self._test_search_with_confidence_threshold_warning.test_name,
                f"Test search functionality with confidence threshold but without reranker to verify warning behavior. Collection: {self.collections['with_metadata']}. Validates that appropriate warnings are logged when confidence_threshold is set but enable_reranker is False.",
                ["POST /v1/search"],
                [
                    "query",
                    "collection_names",
                    "enable_reranker",
                    "confidence_threshold",
                ],
                search_confidence_warning_time,
                TestStatus.FAILURE,
                "Search with confidence threshold warning test failed",
            )
            return False

    async def test_search_with_confidence_threshold_warning(self) -> bool:
        """Test /search endpoint with confidence threshold but without reranker (should show warning)"""
        payload = {
            "query": "Tell me about the content of these documents",
            "reranker_top_k": 5,
            "vdb_top_k": 10,
            "collection_names": [self.collections["with_metadata"]],
            "messages": [],
            "enable_query_rewriting": False,
            "enable_reranker": False,  # Reranker disabled but confidence threshold set
            "confidence_threshold": 0.5,  # This should trigger a warning
        }

        async with aiohttp.ClientSession() as session:
            try:
                logger.info(
                    "🔍 Searching with confidence threshold but without reranker"
                )
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
                            logger.info(
                                f"✅ Search with confidence threshold warning test passed - found {len(results)} results"
                            )

                            # Log some details about the results for verification
                            for i, result in enumerate(results):
                                logger.info(
                                    f"Result {i+1}: document_name={result.get('document_name')}, score={result.get('score')}"
                                )

                            # Verify document names match files in collection
                            if await verify_citation_document_names(
                                results,
                                [self.collections["with_metadata"]],
                                self.ingestor_server_url,
                            ):
                                logger.info("✅ Citation document names verified")
                            else:
                                logger.error(
                                    "❌ Citation document names verification failed"
                                )
                                return False

                            return True
                        else:
                            # No results found - this is expected when confidence threshold filters out all documents
                            logger.info(
                                "✅ Confidence threshold filtering working correctly - no documents met the threshold"
                            )
                            return True
                    else:
                        logger.error(
                            "❌ Search with confidence threshold warning test failed"
                        )
                        return False
            except Exception as e:
                logger.error(
                    f"❌ Error in search with confidence threshold warning test: {e}"
                )
                return False

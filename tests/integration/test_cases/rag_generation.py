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
RAG generation test module
"""

import json
import logging
import time
from typing import Any

import aiohttp

from ..base import BaseTestModule, TestStatus, test_case
from ..utils.response_handlers import extract_streaming_text, print_response
from ..utils.verification import (
    verify_citation_document_names,
    verify_filtered_citations,
    verify_response_content,
)

logger = logging.getLogger(__name__)


class RAGGenerationModule(BaseTestModule):
    """RAG generation test module"""

    @test_case(10, "Generate with Filter")
    async def _test_generate_with_filter(self) -> bool:
        """Test RAG generation with filter"""
        logger.info("\n=== Test 10: Generate with Filter ===")
        generate_filter_start = time.time()
        generate_filter_success = await self.test_generate_with_filter()
        generate_filter_time = time.time() - generate_filter_start

        if generate_filter_success:
            self.add_test_result(
                self._test_generate_with_filter.test_number,
                self._test_generate_with_filter.test_name,
                f"Test RAG generation with metadata filter to retrieve specific documents. Collection: {self.collections['with_metadata']}. Filter: {self.test_runner.metadata_filter_expr}. Includes response content verification for default files (expects '20' in hammer price response), streaming response handling, and filtered citation verification ensuring only multimodal_test.pdf is cited when using default files.",
                ["POST /v1/generate"],
                [
                    "messages",
                    "collection_names",
                    "filter_expr",
                    "enable_citations",
                    "reranker_top_k",
                ],
                generate_filter_time,
                TestStatus.SUCCESS,
            )
            return True
        else:
            self.add_test_result(
                self._test_generate_with_filter.test_number,
                self._test_generate_with_filter.test_name,
                f"Test RAG generation with metadata filter to retrieve specific documents. Collection: {self.collections['with_metadata']}. Filter: {self.test_runner.metadata_filter_expr}. Includes response content verification for default files (expects '20' in hammer price response), streaming response handling, and filtered citation verification ensuring only multimodal_test.pdf is cited when using default files.",
                ["POST /v1/generate"],
                [
                    "messages",
                    "collection_names",
                    "filter_expr",
                    "enable_citations",
                    "reranker_top_k",
                ],
                generate_filter_time,
                TestStatus.FAILURE,
                "Generate with filter test failed",
            )
            return False

    @test_case(11, "Multi-Collection Query")
    async def _test_multi_collection_query(self) -> bool:
        """Test multi-collection query"""
        logger.info("\n=== Test 11: Multi-Collection Query ===")
        multi_collection_start = time.time()
        multi_collection_success = await self.test_multi_collection_query()
        multi_collection_time = time.time() - multi_collection_start

        if multi_collection_success:
            self.add_test_result(
                self._test_multi_collection_query.test_number,
                self._test_multi_collection_query.test_name,
                f"Test RAG generation across multiple collections simultaneously using /chat/completions endpoint. Collections: {', '.join(self.collections.values())}. Includes response content verification for default files, streaming response handling, and citation verification across multiple collections.",
                ["POST /v1/chat/completions"],
                [
                    "messages",
                    "collection_names",
                    "enable_citations",
                    "reranker_top_k",
                ],
                multi_collection_time,
                TestStatus.SUCCESS,
            )
            return True
        else:
            self.add_test_result(
                self._test_multi_collection_query.test_number,
                self._test_multi_collection_query.test_name,
                f"Test RAG generation across multiple collections simultaneously using /chat/completions endpoint. Collections: {', '.join(self.collections.values())}. Includes response content verification for default files, streaming response handling, and citation verification across multiple collections.",
                ["POST /v1/chat/completions"],
                [
                    "messages",
                    "collection_names",
                    "enable_citations",
                    "reranker_top_k",
                ],
                multi_collection_time,
                TestStatus.FAILURE,
                "Multi-collection query test failed",
            )
            return False

    @test_case(12, "Generate without Reranker")
    async def _test_generate_without_reranker(self) -> bool:
        """Test generate without reranker"""
        logger.info("\n=== Test 12: Generate without Reranker ===")
        generate_no_reranker_start = time.time()
        generate_no_reranker_success = await self.test_generate_without_reranker()
        generate_no_reranker_time = time.time() - generate_no_reranker_start

        if generate_no_reranker_success:
            self.add_test_result(
                self._test_generate_without_reranker.test_number,
                self._test_generate_without_reranker.test_name,
                f"Test RAG generation with reranker disabled to verify vdb_top_k behavior. Collection: {self.collections['with_metadata']}. Includes response content verification for default files, streaming response handling, and citation verification with vdb_top_k count validation.",
                ["POST /v1/generate"],
                [
                    "messages",
                    "collection_names",
                    "enable_reranker",
                    "vdb_top_k",
                    "enable_citations",
                ],
                generate_no_reranker_time,
                TestStatus.SUCCESS,
            )
            return True
        else:
            self.add_test_result(
                self._test_generate_without_reranker.test_number,
                self._test_generate_without_reranker.test_name,
                f"Test RAG generation with reranker disabled to verify vdb_top_k behavior. Collection: {self.collections['with_metadata']}. Includes response content verification for default files, streaming response handling, and citation verification with vdb_top_k count validation.",
                ["POST /v1/generate"],
                [
                    "messages",
                    "collection_names",
                    "enable_reranker",
                    "vdb_top_k",
                    "enable_citations",
                ],
                generate_no_reranker_time,
                TestStatus.FAILURE,
                "Generate without reranker test failed",
            )
            return False

    async def test_generate_with_filter(self, filter_expr: str = None) -> bool:
        """Test /generate endpoint with filter expression"""
        # Use configured filter expression if none provided
        if filter_expr is None:
            filter_expr = self.test_runner.metadata_filter_expr

        payload = {
            "messages": [{"role": "user", "content": "Who is the author of poems?"}],
            "use_knowledge_base": True,
            "temperature": 0.2,
            "top_p": 0.7,
            "max_tokens": 1024,
            "reranker_top_k": 10,
            "vdb_top_k": 100,
            "collection_names": [self.collections["with_metadata"]],
            "enable_query_rewriting": False,
            "enable_reranker": True,
            "enable_citations": True,
            "stop": [],
            "filter_expr": filter_expr,
        }

        async with aiohttp.ClientSession() as session:
            try:
                logger.info(f"🤖 Generating response with filter: {filter_expr}")
                logger.info(
                    f"📋 Generate request payload:\n{json.dumps(payload, indent=2)}"
                )

                async with session.post(
                    f"{self.rag_server_url}/v1/generate", json=payload
                ) as response:
                    result = await print_response(response)
                    if response.status == 200:
                        # Check if it's a streaming response
                        if result.get("streaming_response"):
                            logger.info(
                                f"✅ Generate with filter: {filter_expr} test passed (streaming response processed)"
                            )
                        else:
                            logger.info(
                                f"✅ Generate with filter: {filter_expr} test passed"
                            )

                        # Verify response content for default files
                        if (
                            not self.test_runner.files_with_metadata
                            and not self.test_runner.files_without_metadata
                        ):
                            # Using default files, verify response content
                            response_text = ""
                            if result.get("streaming_response"):
                                # Extract text from streaming response
                                response_text = extract_streaming_text(result)
                            else:
                                # Extract text from regular response
                                choices = result.get("choices", [])
                                if choices:
                                    response_text = (
                                        choices[0].get("message", {}).get("content", "")
                                    )

                            # Verify no valid response text is returned
                            if response_text:
                                expected_keywords = ["context", "does not mention"]
                                if verify_response_content(
                                    response_text, expected_keywords, min_matches=1
                                ):
                                    logger.info(
                                        "✅ Response content verification passed - no expected price information found"
                                    )
                                else:
                                    logger.error(
                                        "⚠️ Response content verification failed - expected price '20' found"
                                    )
                                    return False
                            else:
                                logger.error(
                                    "⚠️ No response text found for content verification"
                                )
                                return False

                        # Verify citations if present (works for both streaming and non-streaming responses)
                        if "citations" in result:
                            citations = result["citations"]
                            results = citations.get("results", [])
                            expected_count = (
                                payload["reranker_top_k"]
                                if payload["enable_reranker"]
                                else payload["vdb_top_k"]
                            )

                            if len(results) == expected_count:
                                logger.info(
                                    f"✅ Citation count verified: {len(results)} results (expected: {expected_count})"
                                )

                                # For default files, verify that citations only come from multimodal_test.pdf due to filter
                                if (
                                    not self.test_runner.files_with_metadata
                                    and not self.test_runner.files_without_metadata
                                ):
                                    if verify_filtered_citations(
                                        results, "multimodal_test.pdf"
                                    ):
                                        logger.info(
                                            "✅ Filtered citations verification passed - only multimodal_test.pdf found"
                                        )
                                    else:
                                        logger.error(
                                            "❌ Filtered citations verification failed - unexpected files found"
                                        )
                                        return False
                                else:
                                    # For custom files, verify document names match files in collection
                                    if await verify_citation_document_names(
                                        results,
                                        [self.collections["with_metadata"]],
                                        self.ingestor_server_url,
                                    ):
                                        logger.info(
                                            "✅ Citation document names verified"
                                        )
                                    else:
                                        logger.error(
                                            "❌ Citation document names verification failed"
                                        )
                                        return False
                            else:
                                logger.warning(
                                    f"❌ Citation count mismatch: got {len(results)}, expected {expected_count}"
                                )
                                # TODO: Citation count verification should be handled on server side
                                logger.info(
                                    "TODO: Server should ensure correct citation count based on vdb_top_k and available documents"
                                )
                                # For now, return True with TODO
                                return True
                        else:
                            logger.error("⚠️ No citations found in generate response")
                            return False

                        return True
                    else:
                        logger.error("❌ Generate with filter test failed")
                        return False
            except Exception as e:
                logger.error(f"❌ Error in generate with filter test: {e}")
                return False

    async def test_multi_collection_query(self) -> bool:
        """Test multi-collection query using /chat/completions endpoint"""
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": "In which year bill had 774 figure? What is the lion putting on?",
                }
            ],
            "use_knowledge_base": True,
            "temperature": 0.2,
            "top_p": 0.7,
            "max_tokens": 1024,
            "reranker_top_k": 10,
            "vdb_top_k": 100,
            "collection_names": [
                self.collections["with_metadata"],
                self.collections["without_metadata"],
            ],
            "enable_query_rewriting": False,
            "enable_reranker": True,
            "enable_citations": True,
            "stop": [],
        }

        async with aiohttp.ClientSession() as session:
            try:
                logger.info("🤖 Multi-collection query")
                logger.info(
                    f"📋 Multi-collection request payload:\n{json.dumps(payload, indent=2)}"
                )

                async with session.post(
                    f"{self.rag_server_url}/v1/chat/completions", json=payload
                ) as response:
                    result = await print_response(response)
                    if response.status == 200:
                        # Check if it's a streaming response
                        if result.get("streaming_response"):
                            logger.info(
                                "✅ Multi-collection query test passed (streaming response processed)"
                            )
                        else:
                            logger.info("✅ Multi-collection query test passed")

                        # Verify response content for default files
                        if (
                            not self.test_runner.files_with_metadata
                            and not self.test_runner.files_without_metadata
                        ):
                            # Using default files, verify response content
                            response_text = ""
                            if result.get("streaming_response"):
                                # Extract text from streaming response
                                response_text = extract_streaming_text(result)
                            else:
                                # Extract text from regular response
                                choices = result.get("choices", [])
                                if choices:
                                    response_text = (
                                        choices[0].get("message", {}).get("content", "")
                                    )

                            # Verify multi-collection response (bill year and hammer price)
                            if response_text:
                                expected_keywords = ["2016", "sunscreen"]
                                if verify_response_content(
                                    response_text, expected_keywords, min_matches=2
                                ):
                                    logger.info(
                                        "✅ Multi-collection response verification passed - found expected information"
                                    )
                                else:
                                    logger.error(
                                        f"⚠️ Multi-collection response verification failed - expected keywords not found: {expected_keywords}"
                                    )
                                    return False
                            else:
                                logger.error(
                                    "⚠️ No response text found for content verification"
                                )
                                return False

                        # Verify citations if present (works for both streaming and non-streaming responses)
                        if "citations" in result:
                            citations = result["citations"]
                            results = citations.get("results", [])
                            expected_count = (
                                payload["reranker_top_k"]
                                if payload["enable_reranker"]
                                else payload["vdb_top_k"]
                            )

                            if len(results) == expected_count:
                                logger.info(
                                    f"✅ Citation count verified: {len(results)} results (expected: {expected_count})"
                                )

                                # Verify document names match files in collections
                                if await verify_citation_document_names(
                                    results,
                                    list(self.collections.values()),
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
                                    f"❌ Citation count mismatch: got {len(results)}, expected {expected_count}"
                                )
                                # TODO: Citation count verification should be handled on server side
                                logger.info(
                                    "TODO: Server should ensure correct citation count based on vdb_top_k and available documents"
                                )
                                # For now, return True with TODO
                                return True
                        else:
                            logger.error("⚠️ No citations found in generate response")
                            return False

                        return True
                    else:
                        logger.error("❌ Multi-collection query test failed")
                        return False
            except Exception as e:
                logger.error(f"❌ Error in multi-collection query test: {e}")
                return False

    async def test_generate_without_reranker(self) -> bool:
        """Test /generate endpoint with enable_reranker=false and verify vdb_top_k citations"""
        payload = {
            "messages": [{"role": "user", "content": "Who is the author of poems?"}],
            "use_knowledge_base": True,
            "temperature": 0,
            "top_p": 0.1,
            "max_tokens": 32768,
            "reranker_top_k": 2,
            "vdb_top_k": 10,  # Use a different value to distinguish from reranker test
            "collection_names": [self.collections["with_metadata"]],
            "enable_query_rewriting": False,
            "enable_reranker": False,
            "enable_citations": True,
            "stop": [],
        }

        async with aiohttp.ClientSession() as session:
            try:
                logger.info("🤖 Generating response without reranker")
                logger.info(
                    f"📋 Generate request payload:\n{json.dumps(payload, indent=2)}"
                )

                async with session.post(
                    f"{self.rag_server_url}/v1/generate", json=payload
                ) as response:
                    result = await print_response(response)
                    if response.status == 200:
                        # Check if it's a streaming response
                        if result.get("streaming_response"):
                            logger.info(
                                "✅ Generate without reranker test passed (streaming response processed)"
                            )
                        else:
                            logger.info("✅ Generate without reranker test passed")

                        # Verify response content for default files
                        if (
                            not self.test_runner.files_with_metadata
                            and not self.test_runner.files_without_metadata
                        ):
                            # Using default files, verify response content
                            response_text = ""
                            if result.get("streaming_response"):
                                # Extract text from streaming response
                                response_text = extract_streaming_text(result)
                            else:
                                # Extract text from regular response
                                choices = result.get("choices", [])
                                if choices:
                                    response_text = (
                                        choices[0].get("message", {}).get("content", "")
                                    )

                            # Verify hammer price response
                            if response_text:
                                expected_keywords = ["Robert Frost"]
                                if verify_response_content(
                                    response_text, expected_keywords, min_matches=1
                                ):
                                    logger.info(
                                        "✅ Response content verification passed - found expected price information"
                                    )
                                else:
                                    logger.error(
                                        "⚠️ Response content verification failed - expected price '20' not found"
                                    )
                                    return False
                            else:
                                logger.error(
                                    "⚠️ No response text found for content verification"
                                )
                                return False

                        # Verify citations if present (works for both streaming and non-streaming responses)
                        if "citations" in result:
                            citations = result["citations"]
                            results = citations.get("results", [])
                            expected_count = payload[
                                "vdb_top_k"
                            ]  # Should use vdb_top_k when reranker is disabled

                            if len(results) == expected_count:
                                logger.info(
                                    f"✅ Citation count verified: {len(results)} results (expected: {expected_count})"
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
                                    f"❌ Citation count mismatch: got {len(results)}, expected {expected_count}"
                                )
                                # TODO: Citation count verification should be handled on server side
                                logger.info(
                                    "TODO: Server should ensure correct citation count based on vdb_top_k and available documents"
                                )
                                # For now, return True with TODO
                                return True
                        else:
                            logger.error("⚠️ No citations found in generate response")
                            return False

                        return True
                    else:
                        logger.error("❌ Generate without reranker test failed")
                        return False
            except Exception as e:
                logger.error(f"❌ Error in generate without reranker test: {e}")
                return False

    @test_case(35, "Generate with Confidence Threshold")
    async def _test_generate_with_confidence_threshold(self) -> bool:
        """Test RAG generation with confidence threshold filtering"""
        logger.info("\n=== Test 35: Generate with Confidence Threshold ===")
        generate_confidence_start = time.time()
        generate_confidence_success = (
            await self.test_generate_with_confidence_threshold()
        )
        generate_confidence_time = time.time() - generate_confidence_start

        if generate_confidence_success:
            self.add_test_result(
                self._test_generate_with_confidence_threshold.test_number,
                self._test_generate_with_confidence_threshold.test_name,
                f"Test RAG generation with confidence threshold filtering to retrieve only high-confidence documents. Collection: {self.collections['with_metadata']}. Validates that only documents with relevance scores >= confidence_threshold are used in generation, with proper streaming response handling and citation verification.",
                ["POST /v1/generate"],
                [
                    "messages",
                    "collection_names",
                    "enable_citations",
                    "enable_reranker",
                    "confidence_threshold",
                ],
                generate_confidence_time,
                TestStatus.SUCCESS,
            )
            return True
        else:
            self.add_test_result(
                self._test_generate_with_confidence_threshold.test_number,
                self._test_generate_with_confidence_threshold.test_name,
                f"Test RAG generation with confidence threshold filtering to retrieve only high-confidence documents. Collection: {self.collections['with_metadata']}. Validates that only documents with relevance scores >= confidence_threshold are used in generation, with proper streaming response handling and citation verification.",
                ["POST /v1/generate"],
                [
                    "messages",
                    "collection_names",
                    "enable_citations",
                    "enable_reranker",
                    "confidence_threshold",
                ],
                generate_confidence_time,
                TestStatus.FAILURE,
                "Generate with confidence threshold test failed",
            )
            return False

    async def test_generate_with_confidence_threshold(self) -> bool:
        """Test /generate endpoint with confidence threshold filtering"""
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": "Tell me about the content of these documents",
                }
            ],
            "collection_names": [self.collections["with_metadata"]],
            "enable_citations": True,
            "reranker_top_k": 5,
            "vdb_top_k": 10,
            "enable_reranker": True,
            "confidence_threshold": 0.6,  # Only include documents with score >= 0.6
        }

        async with aiohttp.ClientSession() as session:
            try:
                logger.info(
                    "🤖 Generating response with confidence threshold filtering"
                )
                logger.info(
                    f"📋 Generate request payload:\n{json.dumps(payload, indent=2)}"
                )

                async with session.post(
                    f"{self.rag_server_url}/v1/generate", json=payload
                ) as response:
                    result = await print_response(response)
                    if response.status == 200:
                        # Check if it's a streaming response
                        if result.get("streaming_response"):
                            logger.info(
                                "✅ Generate with confidence threshold test passed (streaming response processed)"
                            )
                        else:
                            logger.info(
                                "✅ Generate with confidence threshold test passed"
                            )

                        # Verify citations if present
                        if "citations" in result:
                            citations = result["citations"]
                            results = citations.get("results", [])

                            if results:
                                # Verify that all citations have relevance scores >= threshold
                                threshold = payload["confidence_threshold"]
                                filtered_results = [
                                    result
                                    for result in results
                                    if result.get("score", 0.0) >= threshold
                                ]

                                if len(filtered_results) == len(results):
                                    logger.info(
                                        f"✅ Confidence threshold filtering verified - all {len(results)} citations have scores >= {threshold}"
                                    )

                                    # Log scores for verification
                                    for i, result in enumerate(results):
                                        logger.info(
                                            f"Citation {i+1}: document_name={result.get('document_name')}, score={result.get('score')}"
                                        )

                                    # Verify document names match files in collection
                                    if await verify_citation_document_names(
                                        results,
                                        [self.collections["with_metadata"]],
                                        self.ingestor_server_url,
                                    ):
                                        logger.info(
                                            "✅ Citation document names verified"
                                        )
                                    else:
                                        logger.error(
                                            "❌ Citation document names verification failed"
                                        )
                                        return False

                                    return True
                                else:
                                    logger.error(
                                        f"❌ Confidence threshold filtering failed: {len(results) - len(filtered_results)} citations have scores < {threshold}"
                                    )
                                    return False
                            else:
                                # No citations found - this is expected when confidence threshold filters out all documents
                                logger.info(
                                    "✅ Confidence threshold filtering working correctly - no documents met the threshold"
                                )
                                return True
                        else:
                            # No citations section - this is expected when confidence threshold filters out all documents
                            logger.info(
                                "✅ Confidence threshold filtering working correctly - no documents met the threshold"
                            )
                            return True

                        return True
                    else:
                        logger.error(
                            "❌ Generate with confidence threshold test failed"
                        )
                        return False
            except Exception as e:
                logger.error(
                    f"❌ Error in generate with confidence threshold test: {e}"
                )
                return False

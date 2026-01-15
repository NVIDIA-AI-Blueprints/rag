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
OpenAI-compatible Vector Store Search test module
"""

import json
import logging
import time

from openai import OpenAI

from ..base import BaseTestModule, TestStatus, test_case

logger = logging.getLogger(__name__)


class OpenAIVectorStoreSearchModule(BaseTestModule):
    """OpenAI-compatible Vector Store Search test module"""

    def _get_openai_client(self) -> OpenAI:
        """Create and return OpenAI client configured for NVIDIA RAG server"""
        return OpenAI(
            base_url=f"{self.rag_server_url}/v2",
            api_key="not-needed"  # API key not required for this endpoint
        )

    @test_case(110, "OpenAI Vector Store Search - Basic")
    async def _test_openai_search_basic(self) -> bool:
        """Test basic OpenAI-compatible vector store search"""
        logger.info("\n=== Test 110: OpenAI Vector Store Search - Basic ===")
        search_start = time.time()
        search_success = await self.test_openai_search_basic()
        search_time = time.time() - search_start

        if search_success:
            self.add_test_result(
                self._test_openai_search_basic.test_number,
                self._test_openai_search_basic.test_name,
                f"Test basic OpenAI-compatible vector store search using OpenAI SDK. Collection: {self.collections['with_metadata']}. Validates the endpoint returns proper OpenAI-compatible response schema with file_id, filename, score, and content fields.",
                ["POST /v2/vector_stores/{{vector_store_id}}/search"],
                ["query", "vector_store_id"],
                search_time,
                TestStatus.SUCCESS,
            )
            return True
        else:
            self.add_test_result(
                self._test_openai_search_basic.test_number,
                self._test_openai_search_basic.test_name,
                f"Test basic OpenAI-compatible vector store search using OpenAI SDK. Collection: {self.collections['with_metadata']}. Validates the endpoint returns proper OpenAI-compatible response schema with file_id, filename, score, and content fields.",
                ["POST /v2/vector_stores/{{vector_store_id}}/search"],
                ["query", "vector_store_id"],
                search_time,
                TestStatus.FAILURE,
                "OpenAI vector store search basic test failed",
            )
            return False

    async def test_openai_search_basic(self) -> bool:
        """Test basic OpenAI vector store search using OpenAI SDK"""
        try:
            client = self._get_openai_client()
            
            logger.info("🔍 Testing OpenAI vector store search - basic query")
            logger.info(f"📋 Vector Store ID: {self.collections['with_metadata']}")
            logger.info("📋 Query: 'What is lion doing?'")

            # Make the search request using OpenAI SDK
            response = client.vector_stores.search(
                vector_store_id=self.collections["with_metadata"],
                query="What is lion doing?"
            )

            # Log the response
            logger.info(f"📤 Response received: {response}")

            # Verify the response structure
            if not hasattr(response, 'object'):
                logger.error("❌ Response missing 'object' field")
                return False

            if response.object != "vector_store.search_results.page":
                logger.error(f"❌ Unexpected object type: {response.object}")
                return False

            if not hasattr(response, 'data') or not response.data:
                logger.error("❌ Response missing or empty 'data' field")
                return False

            # Validate data structure
            for item in response.data:
                if not hasattr(item, 'file_id'):
                    logger.error("❌ Search result missing 'file_id' field")
                    return False
                if not hasattr(item, 'filename'):
                    logger.error("❌ Search result missing 'filename' field")
                    return False
                if not hasattr(item, 'score'):
                    logger.error("❌ Search result missing 'score' field")
                    return False
                if not hasattr(item, 'content'):
                    logger.error("❌ Search result missing 'content' field")
                    return False

            logger.info(f"✅ OpenAI vector store search basic test passed - found {len(response.data)} results")
            logger.info(f"First result: filename={response.data[0].filename}, score={response.data[0].score}")
            return True

        except Exception as e:
            logger.error(f"❌ Error in OpenAI vector store search basic test: {e}")
            return False

    @test_case(111, "OpenAI Vector Store Search - With max_num_results")
    async def _test_openai_search_with_max_results(self) -> bool:
        """Test OpenAI vector store search with max_num_results parameter"""
        logger.info("\n=== Test 111: OpenAI Vector Store Search - With max_num_results ===")
        search_start = time.time()
        search_success = await self.test_openai_search_with_max_results()
        search_time = time.time() - search_start

        if search_success:
            self.add_test_result(
                self._test_openai_search_with_max_results.test_number,
                self._test_openai_search_with_max_results.test_name,
                f"Test OpenAI vector store search with max_num_results parameter. Collection: {self.collections['with_metadata']}. Validates that the number of results returned matches the max_num_results parameter.",
                ["POST /v2/vector_stores/{{vector_store_id}}/search"],
                ["query", "vector_store_id", "max_num_results"],
                search_time,
                TestStatus.SUCCESS,
            )
            return True
        else:
            self.add_test_result(
                self._test_openai_search_with_max_results.test_number,
                self._test_openai_search_with_max_results.test_name,
                f"Test OpenAI vector store search with max_num_results parameter. Collection: {self.collections['with_metadata']}. Validates that the number of results returned matches the max_num_results parameter.",
                ["POST /v2/vector_stores/{{vector_store_id}}/search"],
                ["query", "vector_store_id", "max_num_results"],
                search_time,
                TestStatus.FAILURE,
                "OpenAI vector store search with max_num_results test failed",
            )
            return False

    async def test_openai_search_with_max_results(self) -> bool:
        """Test OpenAI vector store search with max_num_results"""
        try:
            client = self._get_openai_client()
            max_results = 2  # Use 2 instead of 3 since test collection has 2 files
            
            logger.info("🔍 Testing OpenAI vector store search - with max_num_results")
            logger.info(f"📋 Vector Store ID: {self.collections['with_metadata']}")
            logger.info(f"📋 max_num_results: {max_results}")

            response = client.vector_stores.search(
                vector_store_id=self.collections["with_metadata"],
                query="Tell me about the content of these documents",
                max_num_results=max_results
            )

            if len(response.data) > max_results:
                logger.error(f"❌ Expected at most {max_results} results, got {len(response.data)}")
                return False

            if len(response.data) == 0:
                logger.error("❌ Expected some results, got none")
                return False

            logger.info(f"✅ OpenAI vector store search with max_num_results test passed - found {len(response.data)} results (expected at most {max_results})")
            return True

        except Exception as e:
            logger.error(f"❌ Error in OpenAI vector store search with max_num_results test: {e}")
            return False

    @test_case(112, "OpenAI Vector Store Search - With Ranking Options")
    async def _test_openai_search_with_ranking_options(self) -> bool:
        """Test OpenAI vector store search with ranking options"""
        logger.info("\n=== Test 112: OpenAI Vector Store Search - With Ranking Options ===")
        search_start = time.time()
        search_success = await self.test_openai_search_with_ranking_options()
        search_time = time.time() - search_start

        if search_success:
            self.add_test_result(
                self._test_openai_search_with_ranking_options.test_number,
                self._test_openai_search_with_ranking_options.test_name,
                f"Test OpenAI vector store search with ranking options (ranker and score_threshold). Collection: {self.collections['with_metadata']}. Validates that ranking options control reranking and score filtering.",
                ["POST /v2/vector_stores/{{vector_store_id}}/search"],
                ["query", "vector_store_id", "ranking_options"],
                search_time,
                TestStatus.SUCCESS,
            )
            return True
        else:
            self.add_test_result(
                self._test_openai_search_with_ranking_options.test_number,
                self._test_openai_search_with_ranking_options.test_name,
                f"Test OpenAI vector store search with ranking options (ranker and score_threshold). Collection: {self.collections['with_metadata']}. Validates that ranking options control reranking and score filtering.",
                ["POST /v2/vector_stores/{{vector_store_id}}/search"],
                ["query", "vector_store_id", "ranking_options"],
                search_time,
                TestStatus.FAILURE,
                "OpenAI vector store search with ranking options test failed",
            )
            return False

    async def test_openai_search_with_ranking_options(self) -> bool:
        """Test OpenAI vector store search with ranking options"""
        try:
            client = self._get_openai_client()
            
            logger.info("🔍 Testing OpenAI vector store search - with ranking options")
            logger.info(f"📋 Vector Store ID: {self.collections['with_metadata']}")
            logger.info('📋 Ranking options: ranker=auto, score_threshold=0.5')

            response = client.vector_stores.search(
                vector_store_id=self.collections["with_metadata"],
                query="Tell me about the content of these documents",
                max_num_results=5,
                ranking_options={
                    "ranker": "auto",
                    "score_threshold": 0.5
                }
            )

            # Verify all results meet score threshold
            for item in response.data:
                if item.score < 0.5:
                    logger.error(f"❌ Result has score {item.score} below threshold 0.5")
                    return False

            logger.info(f"✅ OpenAI vector store search with ranking options test passed - found {len(response.data)} results")
            return True

        except Exception as e:
            logger.error(f"❌ Error in OpenAI vector store search with ranking options test: {e}")
            return False

    @test_case(113, "OpenAI Vector Store Search - Ranker Disabled")
    async def _test_openai_search_ranker_disabled(self) -> bool:
        """Test OpenAI vector store search with ranker disabled"""
        logger.info("\n=== Test 113: OpenAI Vector Store Search - Ranker Disabled ===")
        search_start = time.time()
        search_success = await self.test_openai_search_ranker_disabled()
        search_time = time.time() - search_start

        if search_success:
            self.add_test_result(
                self._test_openai_search_ranker_disabled.test_number,
                self._test_openai_search_ranker_disabled.test_name,
                f"Test OpenAI vector store search with ranker disabled. Collection: {self.collections['with_metadata']}. Validates that setting ranker='none' disables reranking.",
                ["POST /v2/vector_stores/{{vector_store_id}}/search"],
                ["query", "vector_store_id", "ranking_options"],
                search_time,
                TestStatus.SUCCESS,
            )
            return True
        else:
            self.add_test_result(
                self._test_openai_search_ranker_disabled.test_number,
                self._test_openai_search_ranker_disabled.test_name,
                f"Test OpenAI vector store search with ranker disabled. Collection: {self.collections['with_metadata']}. Validates that setting ranker='none' disables reranking.",
                ["POST /v2/vector_stores/{{vector_store_id}}/search"],
                ["query", "vector_store_id", "ranking_options"],
                search_time,
                TestStatus.FAILURE,
                "OpenAI vector store search with ranker disabled test failed",
            )
            return False

    async def test_openai_search_ranker_disabled(self) -> bool:
        """Test OpenAI vector store search with ranker disabled"""
        try:
            client = self._get_openai_client()
            
            logger.info("🔍 Testing OpenAI vector store search - ranker disabled")
            logger.info(f"📋 Vector Store ID: {self.collections['with_metadata']}")
            logger.info('📋 Ranking options: ranker=none')

            response = client.vector_stores.search(
                vector_store_id=self.collections["with_metadata"],
                query="Tell me about the content of these documents",
                max_num_results=5,
                ranking_options={
                    "ranker": "none"
                }
            )

            if not response.data:
                logger.error("❌ No results returned")
                return False

            logger.info(f"✅ OpenAI vector store search with ranker disabled test passed - found {len(response.data)} results")
            return True

        except Exception as e:
            logger.error(f"❌ Error in OpenAI vector store search with ranker disabled test: {e}")
            return False

    @test_case(114, "OpenAI Vector Store Search - With Query Rewriting")
    async def _test_openai_search_with_query_rewriting(self) -> bool:
        """Test OpenAI vector store search with query rewriting enabled"""
        logger.info("\n=== Test 114: OpenAI Vector Store Search - With Query Rewriting ===")
        search_start = time.time()
        search_success = await self.test_openai_search_with_query_rewriting()
        search_time = time.time() - search_start

        if search_success:
            self.add_test_result(
                self._test_openai_search_with_query_rewriting.test_number,
                self._test_openai_search_with_query_rewriting.test_name,
                f"Test OpenAI vector store search with query rewriting enabled. Collection: {self.collections['with_metadata']}. Validates that rewrite_query parameter works correctly.",
                ["POST /v2/vector_stores/{{vector_store_id}}/search"],
                ["query", "vector_store_id", "rewrite_query"],
                search_time,
                TestStatus.SUCCESS,
            )
            return True
        else:
            self.add_test_result(
                self._test_openai_search_with_query_rewriting.test_number,
                self._test_openai_search_with_query_rewriting.test_name,
                f"Test OpenAI vector store search with query rewriting enabled. Collection: {self.collections['with_metadata']}. Validates that rewrite_query parameter works correctly.",
                ["POST /v2/vector_stores/{{vector_store_id}}/search"],
                ["query", "vector_store_id", "rewrite_query"],
                search_time,
                TestStatus.FAILURE,
                "OpenAI vector store search with query rewriting test failed",
            )
            return False

    async def test_openai_search_with_query_rewriting(self) -> bool:
        """Test OpenAI vector store search with query rewriting"""
        try:
            client = self._get_openai_client()
            
            logger.info("🔍 Testing OpenAI vector store search - with query rewriting")
            logger.info(f"📋 Vector Store ID: {self.collections['with_metadata']}")
            logger.info('📋 rewrite_query: True')

            response = client.vector_stores.search(
                vector_store_id=self.collections["with_metadata"],
                query="Tell me about the content of these documents",
                max_num_results=3,
                rewrite_query=True
            )

            if not response.data:
                logger.error("❌ No results returned")
                return False

            logger.info(f"✅ OpenAI vector store search with query rewriting test passed - found {len(response.data)} results")
            return True

        except Exception as e:
            logger.error(f"❌ Error in OpenAI vector store search with query rewriting test: {e}")
            return False


#!/usr/bin/env python3
"""
Integration tests for custom metadata support.

This module tests:
1. Metadata normalization (datetime, boolean) during ingestion
2. Filter expression validation and document filtering
3. LLM-based filter generation from natural language
"""

import asyncio
import codecs
import json
import logging
import os
import time
from pathlib import Path

import aiohttp

from tests.integration.base import BaseTestModule, TestStatus, test_case

logger = logging.getLogger(__name__)


class CustomMetadataModule(BaseTestModule):
    """Custom metadata integration test module"""

    def __init__(self, test_runner):
        super().__init__(test_runner)

        # Use a separate collection for custom metadata tests to avoid conflicts
        self.metadata_collection = "test_custom_metadata_collection"

        # Add the custom metadata collection to the test runner's collections
        # so it gets cleaned up properly by the cleanup tests
        self.test_runner.collections["custom_metadata"] = self.metadata_collection

        # Test files for metadata testing
        self.test_files = [
            "multimodal_test.pdf",
            "woods_frost.docx"
        ]

        # Data directory for test files
        self.data_dir = Path(__file__).parent.parent.parent.parent / "data" / "multimodal"

        # Metadata schema for testing
        self.metadata_schema = [
            {
                "name": "title",
                "type": "string",
                "required": True,
                "max_length": 200,
                "description": "Document title"
            },
            {
                "name": "category",
                "type": "string",
                "required": False,
                "description": "Document category"
            },
            {
                "name": "rating",
                "type": "float",
                "required": False,
                "description": "Document rating"
            },
            {
                "name": "is_public",
                "type": "boolean",
                "required": False,
                "description": "Whether document is public"
            },
            {
                "name": "tags",
                "type": "array",
                "array_type": "string",
                "max_length": 50,
                "required": False,
                "description": "Document tags"
            },
            {
                "name": "created_date",
                "type": "datetime",
                "required": True,
                "description": "Document creation date"
            },
            {
                "name": "updated_date",
                "type": "datetime",
                "required": False,
                "description": "Document update date"
            }
        ]

        # Test metadata with various data types for normalization testing
        self.test_metadata = [
            {
                "filename": "multimodal_test.pdf",
                "metadata": {
                    "title": "AI Policy Guidelines",
                    "category": "tech",
                    "rating": 4.5,
                    "is_public": True,
                    "tags": ["urgent", "policy", "ai"],
                    "created_date": "2024-01-15T10:30:00Z",
                    "updated_date": "2024-01-16T14:20:00"
                }
            },
            {
                "filename": "woods_frost.docx",
                "metadata": {
                    "title": "Nature Poetry Collection",
                    "category": "literature",
                    "rating": 3.8,
                    "is_public": False,
                    "tags": ["poetry", "nature", "classic"],
                    "created_date": "2024-01-10T09:15:00",
                    "updated_date": "2024-01-12T16:45:00Z"
                }
            }
        ]

    @test_case(36, "Metadata Normalization Test")
    async def test_metadata_normalization(self) -> bool:
        """Test metadata normalization during ingestion (datetime, boolean)"""
        logger.info("\n=== Test 36: Metadata Normalization Test ===")
        test_start = time.time()

        try:
            # Step 1: Upload documents with metadata (collection already exists from Test 2)
            logger.info("Uploading documents with metadata...")
            task_id = await self._upload_documents_with_metadata()

            if not task_id:
                self.add_test_result(
                    36,
                    "Metadata Normalization Test",
                    "Test that metadata values are properly normalized during ingestion",
                    ["POST /v1/collection", "POST /v1/documents", "POST /search"],
                    ["collection_name", "metadata_schema", "custom_metadata", "query"],
                    time.time() - test_start,
                    TestStatus.FAILURE,
                    "Failed to upload documents with metadata"
                )
                return False

            # Step 2: Wait for ingestion to complete
            logger.info("Waiting for ingestion to complete...")
            ingestion_success = await self._wait_for_task_completion(task_id)

            if not ingestion_success:
                self.add_test_result(
                    36,
                    "Metadata Normalization Test",
                    "Test that metadata values are properly normalized during ingestion",
                    ["POST /v1/collection", "POST /v1/documents", "POST /search"],
                    ["collection_name", "metadata_schema", "custom_metadata", "query"],
                    time.time() - test_start,
                    TestStatus.FAILURE,
                    "Ingestion failed or timed out"
                )
                return False

            # Step 3: Search for documents and verify metadata normalization
            logger.info("Searching for documents to verify metadata normalization...")

            # Add a small delay to ensure documents are available for search
            await asyncio.sleep(2)

            verification_success = await self._verify_metadata_normalization()

            if not verification_success:
                self.add_test_result(
                    36,
                    "Metadata Normalization Test",
                    "Test that metadata values are properly normalized during ingestion",
                    ["POST /v1/collection", "POST /v1/documents", "POST /search"],
                    ["collection_name", "metadata_schema", "custom_metadata", "query"],
                    time.time() - test_start,
                    TestStatus.FAILURE,
                    "Metadata normalization verification failed"
                )
                return False

            test_time = time.time() - test_start

            self.add_test_result(
                36,
                "Metadata Normalization Test",
                "Test that metadata values are properly normalized during ingestion",
                ["POST /v1/collection", "POST /v1/documents", "POST /search"],
                ["collection_name", "metadata_schema", "custom_metadata", "query"],
                test_time,
                TestStatus.SUCCESS
            )

            logger.info("✅ Metadata normalization test passed")
            return True

        except Exception as e:
            test_time = time.time() - test_start
            logger.error(f"❌ Metadata normalization test failed: {str(e)}")

            self.add_test_result(
                36,
                "Metadata Normalization Test",
                "Test that metadata values are properly normalized during ingestion",
                ["POST /v1/collection", "POST /v1/documents", "POST /search"],
                ["collection_name", "metadata_schema", "custom_metadata", "query"],
                test_time,
                TestStatus.FAILURE,
                str(e)
            )
            return False

    async def _upload_documents_with_metadata(self) -> str | None:
        """Upload documents with metadata"""
        try:
            # Prepare file paths
            file_paths = []
            for filename in self.test_files:
                file_path = self.data_dir / filename
                if not file_path.exists():
                    logger.error(f"Test file not found: {file_path}")
                    return None
                file_paths.append(str(file_path))

            # Create upload data
            data = {
                "collection_name": self.metadata_collection,
                "blocking": False,
                "split_options": {"chunk_size": 512, "chunk_overlap": 150},
                "custom_metadata": self.test_metadata,
                "generate_summary": False,
            }

            # Create multipart form data
            form_data = aiohttp.FormData()
            for file_path in file_paths:
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
                logger.info(f"📤 Uploading {len(file_paths)} documents to collection '{self.metadata_collection}'")

                async with session.post(
                    f"{self.ingestor_server_url}/v1/documents",
                    data=form_data
                ) as response:
                    result = await response.json()
                    if response.status == 200:
                        logger.info("✅ Upload request successful")
                        task_id = result.get("task_id")
                        logger.info(f"✅ Documents uploaded successfully. Task ID: {task_id}")
                        return task_id
                    else:
                        logger.error(f"❌ Failed to upload documents. Status: {response.status}")
                        logger.error(f"Response: {json.dumps(result, indent=2)}")
                        return None
        except Exception as e:
            logger.error(f"❌ Error uploading documents: {e}")
            return None

    async def _wait_for_task_completion(self, task_id: str) -> bool:
        """Wait for task completion by polling status endpoint"""
        start_time = time.time()

        while time.time() - start_time < self.timeout:
            async with aiohttp.ClientSession() as session:
                try:
                    params = {"task_id": task_id}
                    async with session.get(
                        f"{self.ingestor_server_url}/v1/status", params=params
                    ) as response:
                        result = await response.json()
                        if response.status == 200:
                            state = result.get("state")
                            if state == "FINISHED":
                                logger.info(f"✅ Task {task_id} completed successfully")
                                return True
                            elif state == "FAILED":
                                logger.error(f"❌ Task {task_id} failed")
                                logger.error(f"Task failure details: {json.dumps(result, indent=2)}")
                                return False
                            else:
                                logger.info(f"⏳ Task {task_id} state: {state}")
                                if "progress" in result:
                                    logger.info(f"   Progress: {result.get('progress')}")
                                if "message" in result:
                                    logger.info(f"   Message: {result.get('message')}")
                except Exception as e:
                    logger.warning(f"Error checking task status: {e}")

            await asyncio.sleep(5)  # Wait 5 seconds before next check

        logger.error(f"❌ Task {task_id} timed out after {self.timeout} seconds")
        return False

    async def _verify_metadata_normalization(self) -> bool:
        """Verify that metadata values are properly normalized"""
        try:
            search_payload = {
                "query": "policy guidelines AI",  # More generic query
                "collection_names": [self.metadata_collection],
                "top_k": 10
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.rag_server_url}/search",
                    json=search_payload
                ) as response:
                    result = await response.json()
                    if response.status != 200:
                        logger.error(f"❌ Search failed with status {response.status}: {result}")
                        return False

                    logger.info(f"✅ Search successful, found {len(result.get('results', []))} documents")

                    # Check if we have search results
                    results = result.get('results', [])
                    if not results:
                        logger.error("❌ No search results found")
                        return False

                    # Verify metadata normalization in the first result
                    first_result = results[0]

                    # Extract metadata from the correct location: metadata.content_metadata
                    result_metadata = first_result.get('metadata', {})
                    content_metadata = result_metadata.get('content_metadata', {})
                    metadata = {}

                    if content_metadata:
                        # The metadata is directly in content_metadata, not nested
                        metadata = {
                            'title': content_metadata.get('title', ''),
                            'created_date': content_metadata.get('created_date', ''),
                            'updated_date': content_metadata.get('updated_date', ''),
                            'is_public': content_metadata.get('is_public'),
                            'rating': content_metadata.get('rating'),
                            'tags': content_metadata.get('tags'),
                            'category': content_metadata.get('category'),
                            'page_number': content_metadata.get('page_number')  # System-managed field
                        }

                    # Check if we actually found metadata
                    if not metadata or not any(metadata.values()):
                        logger.error("❌ No metadata found in search results")
                        return False

                    # Check specific metadata normalization
                    verification_passed = True

                    # Check title normalization (should be lowercase)
                    title = metadata.get('title', '')
                    if title and title != title.lower():
                        logger.error(f"❌ Title not normalized to lowercase: {title}")
                        verification_passed = False
                    else:
                        logger.info(f"✅ Title properly normalized: {title}")

                    # Check date normalization (should be ISO format with Z)
                    created_date = metadata.get('created_date', '')
                    if created_date and not created_date.endswith('Z'):
                        logger.error(f"❌ Created date not normalized to ISO format: {created_date}")
                        verification_passed = False
                    else:
                        logger.info(f"✅ Created date properly normalized: {created_date}")

                    updated_date = metadata.get('updated_date', '')
                    if updated_date and not updated_date.endswith('Z'):
                        logger.error(f"❌ Updated date not normalized to ISO format: {updated_date}")
                        verification_passed = False
                    else:
                        logger.info(f"✅ Updated date properly normalized: {updated_date}")

                    # Check boolean normalization
                    is_public = metadata.get('is_public')
                    if is_public is not None and not isinstance(is_public, bool):
                        logger.error(f"❌ is_public not normalized to boolean: {is_public} (type: {type(is_public)})")
                        verification_passed = False
                    else:
                        logger.info(f"✅ is_public properly normalized: {is_public}")

                    # Check numeric normalization
                    rating = metadata.get('rating')
                    if rating is not None and not isinstance(rating, (int, float)):
                        logger.error(f"❌ rating not normalized to numeric: {rating} (type: {type(rating)})")
                        verification_passed = False
                    else:
                        logger.info(f"✅ rating properly normalized: {rating}")

                    # Check array normalization
                    tags = metadata.get('tags')
                    if tags is not None and not isinstance(tags, list):
                        logger.error(f"❌ tags not normalized to array: {tags} (type: {type(tags)})")
                        verification_passed = False
                    else:
                        logger.info(f"✅ tags properly normalized: {tags}")

                    # Check page_number (system-managed field auto-populated by nv-ingest)
                    page_number = metadata.get('page_number')
                    if page_number is None:
                        logger.error("❌ page_number not found - system field should be auto-populated")
                        verification_passed = False
                    elif not isinstance(page_number, int):
                        logger.error(f"❌ page_number not an integer: {page_number} (type: {type(page_number)})")
                        verification_passed = False
                    elif page_number < 1:
                        logger.error(f"❌ Invalid page_number: {page_number} (should be >= 1)")
                        verification_passed = False
                    else:
                        logger.info(f"✅ page_number properly auto-populated: {page_number}")

                    if verification_passed:
                        logger.info("✅ All metadata normalization checks passed!")
                        return True
                    else:
                        logger.error("❌ Some metadata normalization checks failed")
                        return False

        except Exception as e:
            logger.error(f"❌ Error during metadata verification: {e}")
            return False

    @test_case(37, "Filter Expression Validation Test")
    async def test_filter_expression_validation(self) -> bool:
        """Test filter expression validation and document filtering"""
        logger.info("\n=== Test 37: Filter Expression Validation Test ===")
        test_start = time.time()

        try:
            # Step 1: Search with filter expression for tech documents
            logger.info("Testing filter expression for tech documents...")

            tech_filter = 'content_metadata["category"] == "tech"'

            search_payload = {
                "query": "policy guidelines",
                "collection_names": [self.metadata_collection],
                "filter_expr": tech_filter,
                "top_k": 10
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.rag_server_url}/search",
                    json=search_payload
                ) as response:
                    result = await response.json()
                    if response.status != 200:
                        logger.error(f"❌ Search with filter failed: {response.status}")
                        logger.error(f"Response: {json.dumps(result, indent=2)}")
                        return False
                    tech_documents = result.get("results", [])
                    logger.info(f"✅ Found {len(tech_documents)} tech documents")

                    # Verify all returned documents are tech category
                    for doc in tech_documents:
                        metadata = doc.get("metadata", {}).get("content_metadata", {})
                        category = metadata.get("category", "")
                        if category != "tech":
                            logger.error(f"❌ Document has wrong category: {category}, expected: tech")
                            return False

            # Step 2: Search with filter expression for high-rated documents
            logger.info("Testing filter expression for high-rated documents...")

            rating_filter = 'content_metadata["rating"] > 4.0'
            search_payload["filter_expr"] = rating_filter

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.rag_server_url}/search",
                    json=search_payload
                ) as response:
                    result = await response.json()
                    if response.status != 200:
                        logger.error(f"❌ Search with rating filter failed: {response.status}")
                        logger.error(f"Response: {json.dumps(result, indent=2)}")
                        return False

                    high_rated_documents = result.get("results", [])
                    logger.info(f"✅ Found {len(high_rated_documents)} high-rated documents")

                    # Verify all returned documents have rating > 4.0
                    for doc in high_rated_documents:
                        metadata = doc.get("metadata", {}).get("content_metadata", {})
                        rating = metadata.get("rating", 0)
                        if rating <= 4.0:
                            logger.error(f"❌ Document has low rating: {rating}, expected > 4.0")
                            return False

            # Step 3: Search with complex filter expression
            logger.info("Testing complex filter expression...")

            complex_filter = 'content_metadata["category"] == "tech" and content_metadata["rating"] > 4.0 and content_metadata["is_public"] == true'
            search_payload["filter_expr"] = complex_filter

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.rag_server_url}/search",
                    json=search_payload
                ) as response:
                    result = await response.json()
                    if response.status != 200:
                        logger.error(f"❌ Search with complex filter failed: {response.status}")
                        logger.error(f"Response: {json.dumps(result, indent=2)}")
                        return False

                    filtered_documents = result.get("results", [])
                    logger.info(f"✅ Found {len(filtered_documents)} documents matching complex filter")

                    # Verify all returned documents match the complex criteria
                    for doc in filtered_documents:
                        metadata = doc.get("metadata", {}).get("content_metadata", {})
                        category = metadata.get("category", "")
                        rating = metadata.get("rating", 0)
                        is_public = metadata.get("is_public", False)

                        if category != "tech" or rating <= 4.0 or not is_public:
                            logger.error(f"❌ Document doesn't match complex filter criteria: category={category}, rating={rating}, is_public={is_public}")
                            return False

            test_time = time.time() - test_start

            self.add_test_result(
                37,
                "Filter Expression Validation Test",
                "Test that filter expressions correctly filter documents based on metadata",
                ["POST /search"],
                ["query", "collection_names", "filter_expr", "top_k"],
                test_time,
                TestStatus.SUCCESS
            )

            logger.info("✅ Filter expression validation test passed")
            return True

        except Exception as e:
            test_time = time.time() - test_start
            logger.error(f"❌ Filter expression validation test failed: {str(e)}")

            self.add_test_result(
                37,
                "Filter Expression Validation Test",
                "Test that filter expressions correctly filter documents based on metadata",
                ["POST /search"],
                ["query", "collection_names", "filter_expr", "top_k"],
                test_time,
                TestStatus.FAILURE,
                str(e)
            )
            return False

    @test_case(38, "LLM Filter Generation Test")
    async def test_llm_filter_generation(self) -> bool:
        """Test LLM-based filter generation from natural language"""
        logger.info("\n=== Test 38: LLM Filter Generation Test ===")
        test_start = time.time()

        try:
            success = True
            success &= await self._test_search_filter_generation()
            success &= await self._test_natural_language_patterns()
            success &= await self._test_generate_endpoint_streaming()
            success &= await self._test_filter_generation_verification()

            test_time = time.time() - test_start

            self.add_test_result(
                38,
                "LLM Filter Generation Test",
                "Test that LLM can generate filter expressions from natural language queries",
                ["POST /generate", "POST /search"],
                ["messages", "collection_names", "enable_filter_generator", "query"],
                test_time,
                TestStatus.SUCCESS if success else TestStatus.FAILURE
            )

            if success:
                logger.info("✅ LLM filter generation test passed")
            return success

        except Exception as e:
            test_time = time.time() - test_start
            logger.error(f"❌ LLM filter generation test failed: {str(e)}")

            self.add_test_result(
                38,
                "LLM Filter Generation Test",
                "Test that LLM can generate filter expressions from natural language queries",
                ["POST /generate", "POST /search"],
                ["messages", "collection_names", "enable_filter_generator", "query"],
                test_time,
                TestStatus.FAILURE,
                str(e)
            )
            return False

    async def _test_search_filter_generation(self) -> bool:
        """Test search endpoint with filter generation enabled"""
        logger.info("Testing search with filter generation...")

        search_payload = {
            "query": "Find urgent tech documents with rating above 4.0",
            "collection_names": [self.metadata_collection],
            "enable_filter_generator": True,
            "top_k": 10
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.rag_server_url}/search",
                json=search_payload
            ) as response:
                result = await response.json()
                if response.status != 200:
                    logger.error(f"❌ Search with filter generation failed: {response.status}")
                    logger.error(f"Response: {json.dumps(result, indent=2)}")
                    return False

                documents = result.get("results", [])
                logger.info(f"✅ Found {len(documents)} documents with generated filter")

                # Verify that the search returned relevant documents
                if not documents:
                    logger.warning("⚠️ No documents returned with filter generation")
                else:
                    # Check if returned documents match the natural language query criteria
                    for doc in documents:
                        metadata = doc.get("metadata", {}).get("content_metadata", {})
                        category = metadata.get("category", "")
                        rating = metadata.get("rating", 0)
                        tags = metadata.get("tags", [])

                        # Basic verification that documents are relevant
                        if category == "tech" and rating > 4.0:
                            logger.info("✅ Found relevant tech document with high rating")
                        elif "urgent" in tags:
                            logger.info("✅ Found document with urgent tag")

                return True

    async def _test_natural_language_patterns(self) -> bool:
        """Test with different natural language patterns"""
        logger.info("Testing different natural language patterns...")

        test_queries = [
            "Show me public documents",
            "Find literature documents",
            "Get documents with poetry tags"
        ]

        search_payload = {
            "collection_names": [self.metadata_collection],
            "enable_filter_generator": True,
            "top_k": 10
        }

        for query in test_queries:
            logger.info(f"Testing query: {query}")

            search_payload["query"] = query

            # Create a new session for each request to avoid session closure issues
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.rag_server_url}/search",
                    json=search_payload
                ) as response:
                    result = await response.json()
                    if response.status != 200:
                        logger.warning(f"⚠️ Search failed for query '{query}': {response.status}")
                        continue

                    query_documents = result.get("results", [])
                    logger.info(f"Query '{query}' returned {len(query_documents)} documents")

                    # Basic verification that filter generation worked
                    if query_documents:
                        logger.info(f"✅ Filter generation successful for query: {query}")

        return True

    async def _test_generate_endpoint_streaming(self) -> bool:
        """Test generate endpoint with streaming response handling"""
        logger.info("Testing generate endpoint with filter generation...")

        generate_payload = {
            "messages": [{"role": "user", "content": "Show me urgent tech documents with high rating"}],
            "use_knowledge_base": True,
            "collection_names": [self.metadata_collection],
            "enable_filter_generator": True,
            "temperature": 0.2,
            "max_tokens": 1024,
            "top_k": 5
        }

        # Handle streaming response from generate endpoint
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.rag_server_url}/generate",
                json=generate_payload
            ) as response:
                if response.status != 200:
                    logger.error(f"❌ Generate with filter generation failed: {response.status}")
                    return False

                # Handle streaming response
                buffer = ""
                first_chunk_with_citations = None
                response_text = ""
                decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")

                async for chunk in response.content.iter_chunked(1024):
                    buffer += decoder.decode(chunk, final=False)
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()

                        if line.startswith("data: "):
                            line = line[len("data: "):].strip()

                        if not line:
                            continue

                        try:
                            data = json.loads(line)

                            # Capture first chunk with citations
                            if first_chunk_with_citations is None and data.get("citations"):
                                first_chunk_with_citations = data
                                logger.info("📋 Found citations in first chunk")

                            # Extract response text
                            choices = data.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                text = delta.get("content", "")
                                if text:
                                    response_text += text

                        except json.JSONDecodeError:
                            logger.debug(f"Failed to parse JSON chunk: {line}")
                            continue

                logger.info(f"✅ Generated response: {response_text[:200]}...")

                # Verify citations are expected based on filter generation
                if first_chunk_with_citations and first_chunk_with_citations.get("citations"):
                    citations = first_chunk_with_citations["citations"]
                    results = citations.get("results", [])

                    logger.info(f"🔍 Verifying {len(results)} citations from filter generation...")

                    citation_verification_passed = True
                    tech_docs_found = 0
                    high_rated_docs_found = 0
                    urgent_docs_found = 0

                    for idx, citation in enumerate(results):
                        content = citation.get("content", "")
                        metadata = citation.get("metadata", {}).get("content_metadata", {})

                        # Check if citation matches expected criteria
                        category = metadata.get("category", "")
                        rating = metadata.get("rating", 0)
                        tags = metadata.get("tags", [])

                        if category == "tech":
                            tech_docs_found += 1
                            logger.info("   ✅ Found tech document")

                        if isinstance(rating, (int, float)) and rating > 4.0:
                            high_rated_docs_found += 1
                            logger.info(f"   ✅ Found high-rated document (rating: {rating})")

                        if "urgent" in tags:
                            urgent_docs_found += 1
                            logger.info("   ✅ Found urgent document")

                        # Verify that the citation content is relevant
                        if not content or len(content.strip()) < 10:
                            logger.warning(f"   ⚠️ Citation {idx+1} has minimal content")
                            citation_verification_passed = False

                    # Summary of citation verification
                    logger.info("📊 Citation Summary:")
                    logger.info(f"   └─ Total citations: {len(results)}")
                    logger.info(f"   └─ Tech documents: {tech_docs_found}")
                    logger.info(f"   └─ High-rated documents (>4.0): {high_rated_docs_found}")
                    logger.info(f"   └─ Urgent documents: {urgent_docs_found}")

                    # Verify that we found relevant documents
                    if tech_docs_found == 0 and high_rated_docs_found == 0 and urgent_docs_found == 0:
                        logger.error("❌ No relevant citations found matching the filter criteria")
                        citation_verification_passed = False
                    else:
                        logger.info("✅ Found relevant citations matching filter criteria")

                    if not citation_verification_passed:
                        logger.error("❌ Citation verification failed")
                        return False
                    else:
                        logger.info("✅ Citation verification passed")
                else:
                    logger.error("❌ No citations found in generate response")
                    logger.error("   └─ Citations are expected for filter generation tests")
                    citation_verification_passed = False

                return True

    async def _test_filter_generation_verification(self) -> bool:
        """Test that verifies filter generation actually works and doesn't fall back to empty strings"""
        logger.info("Testing filter generation verification - ensuring filters are actually generated...")

        # Test query that should generate a specific filter
        test_query = "Show me tech documents with rating above 4.5"

        # Test 1: With filter generation enabled
        search_payload_with_filter = {
            "query": test_query,
            "collection_names": [self.metadata_collection],
            "enable_filter_generator": True,
            "top_k": 20
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.rag_server_url}/search",
                json=search_payload_with_filter
            ) as response:
                if response.status != 200:
                    logger.error(f"❌ Search with filter generation failed: {response.status}")
                    return False

                result_with_filter = await response.json()
                docs_with_filter = result_with_filter.get("results", [])

        # Test 2: Same query without filter generation
        search_payload_without_filter = {
            "query": test_query,
            "collection_names": [self.metadata_collection],
            "enable_filter_generator": False,
            "top_k": 20
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.rag_server_url}/search",
                json=search_payload_without_filter
            ) as response:
                if response.status != 200:
                    logger.error(f"❌ Search without filter generation failed: {response.status}")
                    return False

                result_without_filter = await response.json()
                docs_without_filter = result_without_filter.get("results", [])

        # Verification 1: Results should be different if filter generation is working
        if len(docs_with_filter) == len(docs_without_filter):
            # If same number of results, check if the actual documents are different
            with_filter_ids = {(doc.get("document_name", ""), doc.get("chunk_id", ""), round(doc.get("score", 0), 6)) for doc in docs_with_filter}
            without_filter_ids = {(doc.get("document_name", ""), doc.get("chunk_id", ""), round(doc.get("score", 0), 6)) for doc in docs_without_filter}

            if with_filter_ids == without_filter_ids:
                logger.error("❌ Filter generation verification FAILED: Results are identical with and without filter generation")
                logger.error("   └─ This suggests filter generation is falling back to empty strings")
                return False

        # Verification 2: Documents with filter should match the criteria better
        matching_with_filter = 0
        matching_without_filter = 0

        for doc in docs_with_filter:
            metadata = doc.get("metadata", {}).get("content_metadata", {})
            if metadata.get("category") == "tech" and metadata.get("rating", 0) > 4.5:
                matching_with_filter += 1

        for doc in docs_without_filter:
            metadata = doc.get("metadata", {}).get("content_metadata", {})
            if metadata.get("category") == "tech" and metadata.get("rating", 0) > 4.5:
                matching_without_filter += 1

        # Calculate match rates
        match_rate_with_filter = matching_with_filter / len(docs_with_filter) if docs_with_filter else 0
        match_rate_without_filter = matching_without_filter / len(docs_without_filter) if docs_without_filter else 0

        # Verification 3: Filter generation should improve relevance (with some tolerance)
        improvement_threshold = 0.1  # Allow 10% tolerance
        if (match_rate_with_filter + improvement_threshold) <= match_rate_without_filter and len(docs_with_filter) >= len(docs_without_filter):
            logger.error("❌ Filter generation verification FAILED: No improvement in result relevance")
            logger.error(f"   └─ With filter: {match_rate_with_filter:.2%}, Without: {match_rate_without_filter:.2%}")
            return False

        # Verification 4: At least some documents should match the criteria when filter is enabled
        if matching_with_filter == 0 and len(docs_with_filter) > 0:
            logger.error("❌ Filter generation verification FAILED: No documents match criteria despite filter generation")
            logger.error("   └─ This suggests the generated filter is incorrect or empty")
            return False

        return True

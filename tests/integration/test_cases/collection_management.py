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
Collection management test module
"""

import asyncio
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import aiohttp

from ..base import BaseTestModule, TestStatus, test_case

logger = logging.getLogger(__name__)


class CollectionManagementModule(BaseTestModule):
    """Collection management test module"""

    CUSTOM_METADATA_COLLECTION = "test_custom_metadata_collection"
    TEST_FILE_TYPES_COLLECTION = "test_file_types"
    CSV_DELETION_BATCH_COLLECTION = "test_csv_deletion_batch"
    CATALOG_COLLECTION = "test_catalog_collection"

    @property
    def expected_collections(self):
        """Get all expected collections for this module"""
        return list(self.collections.values()) + [
            self.CUSTOM_METADATA_COLLECTION,
            self.TEST_FILE_TYPES_COLLECTION,
            self.CSV_DELETION_BATCH_COLLECTION,
            self.CATALOG_COLLECTION,
        ]

    @test_case(2, "Create Collections")
    async def _test_create_collections(self) -> bool:
        """Test creating collections"""
        logger.info("\n=== Test 2: Create Collections ===")
        collection_start = time.time()

        # Basic metadata schema for standard collections
        basic_metadata_schema = [
            {
                "name": "timestamp",
                "type": "datetime",
                "description": "Timestamp of when the document was created",
            },
            {
                "name": "meta_field_1",
                "type": "string",
                "description": "Description for the document",
            },
        ]

        # Custom metadata schema for custom metadata tests
        custom_metadata_schema = [
            {
                "name": "title",
                "type": "string",
                "required": True,
                "max_length": 200,
                "description": "Document title",
            },
            {
                "name": "category",
                "type": "string",
                "required": False,
                "description": "Document category",
            },
            {
                "name": "rating",
                "type": "float",
                "required": False,
                "description": "Document rating",
            },
            {
                "name": "is_public",
                "type": "boolean",
                "required": False,
                "description": "Whether document is public",
            },
            {
                "name": "tags",
                "type": "array",
                "array_type": "string",
                "max_length": 50,
                "required": False,
                "description": "Document tags",
            },
            {
                "name": "created_date",
                "type": "datetime",
                "required": True,
                "description": "Document creation date",
            },
            {
                "name": "updated_date",
                "type": "datetime",
                "required": False,
                "description": "Document update date",
            },
        ]

        collection1_success = await self._create_collection(
            self.collections["with_metadata"], basic_metadata_schema
        )
        collection2_success = await self._create_collection(
            self.collections["without_metadata"]
        )
        collection3_success = await self._create_collection(
            self.CUSTOM_METADATA_COLLECTION, custom_metadata_schema
        )
        collection4_success = await self._create_collection(
            self.TEST_FILE_TYPES_COLLECTION
        )
        collection5_success = await self._create_collection(
            self.CSV_DELETION_BATCH_COLLECTION
        )
        # Create catalog collection for Test 71
        collection6_success = await self._create_collection_with_catalog(
            self.CATALOG_COLLECTION,
            description="Test catalog collection",
            tags=["test", "integration"],
            owner="Test Team",
            created_by="integration_test",
            business_domain="Testing",
            status="Active",
        )
        collection_time = time.time() - collection_start

        if (
            collection1_success
            and collection2_success
            and collection3_success
            and collection4_success
            and collection5_success
            and collection6_success
        ):
            self.add_test_result(
                self._test_create_collections.test_number,
                self._test_create_collections.test_name,
                f"Create six test collections - one with basic metadata schema, one without metadata, one with custom metadata schema, one for file type testing, one for CSV deletion batch processing, and one with catalog metadata. Collections: {', '.join(self.expected_collections)}. Basic metadata schema includes fields: timestamp (datetime), meta_field_1 (string). Custom metadata schema includes fields: title, category, rating, is_public, tags, created_date, updated_date. Catalog collection includes: description, tags, owner, created_by, business_domain, status.",
                ["POST /v1/collection"],
                ["collection_name", "embedding_dimension", "metadata_schema"],
                collection_time,
                TestStatus.SUCCESS,
            )
            return True
        else:
            self.add_test_result(
                self._test_create_collections.test_number,
                self._test_create_collections.test_name,
                f"Create six test collections - one with basic metadata schema, one without metadata, one with custom metadata schema, one for file type testing, one for CSV deletion batch processing, and one with catalog metadata. Collections: {', '.join(self.expected_collections)}. Basic metadata schema includes fields: timestamp (datetime), meta_field_1 (string). Custom metadata schema includes fields: title, category, rating, is_public, tags, created_date, updated_date. Catalog collection includes: description, tags, owner, created_by, business_domain, status.",
                ["POST /v1/collection"],
                ["collection_name", "embedding_dimension", "metadata_schema"],
                collection_time,
                TestStatus.FAILURE,
                "Failed to create one or more collections",
            )
            return False

    @test_case(3, "Verify Collections")
    async def _test_verify_collections(self) -> bool:
        """Test verifying collections"""
        logger.info("\n=== Test 3: Verify Collections ===")
        verify_start = time.time()
        verify_success = await self._verify_collections()
        verify_time = time.time() - verify_start

        if verify_success:
            self.add_test_result(
                self._test_verify_collections.test_number,
                self._test_verify_collections.test_name,
                f"Verify collections are created and metadata schema is properly configured. Collections: {', '.join(self.expected_collections)}. Validates metadata schema fields (timestamp: datetime, meta_field_1: string) with type and description verification.",
                ["GET /v1/collections"],
                [
                    "total_collections",
                    "collections[].collection_name",
                    "collections[].metadata_schema",
                ],
                verify_time,
                TestStatus.SUCCESS,
            )
            return True
        else:
            self.add_test_result(
                self._test_verify_collections.test_number,
                self._test_verify_collections.test_name,
                f"Verify collections are created and metadata schema is properly configured. Collections: {', '.join(self.expected_collections)}. Validates metadata schema fields (timestamp: datetime, meta_field_1: string) with type and description verification.",
                ["GET /v1/collections"],
                [
                    "total_collections",
                    "collections[].collection_name",
                    "collections[].metadata_schema",
                ],
                verify_time,
                TestStatus.FAILURE,
                "Collection verification failed",
            )
            return False

    async def _create_collection(
        self, collection_name: str, metadata_schema: list[dict[str, Any]] = None
    ) -> bool:
        """Create a collection with optional metadata schema"""
        try:
            payload = {
                "collection_name": collection_name,
                "embedding_dimension": 2048,
            }

            if metadata_schema:
                payload["metadata_schema"] = metadata_schema

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ingestor_server_url}/v1/collection", json=payload
                ) as response:
                    result = await response.json()
                    if response.status == 200:
                        logger.info(
                            f"✅ Collection '{collection_name}' created successfully:"
                        )
                        logger.info(f"Response JSON:\n{json.dumps(result, indent=2)}")
                        return True
                    else:
                        logger.error(
                            f"❌ Failed to create collection '{collection_name}': {response.status}"
                        )
                        logger.error(f"Response JSON:\n{json.dumps(result, indent=2)}")
                        return False
        except Exception as e:
            logger.error(f"❌ Error creating collection '{collection_name}': {e}")
            return False

    async def _create_collections(self, collection_names: list[str]) -> bool:
        """API to create multiple collections"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ingestor_server_url}/v1/collections", json=collection_names
                ) as response:
                    result = await response.json()
                    if (
                        response.status == 200
                        and result.get("successful") == collection_names
                    ):
                        logger.info(
                            f"✅ Collections '{collection_names}' created successfully:"
                        )
                        logger.info(f"Response JSON:\n{json.dumps(result, indent=2)}")
                        return True
                    else:
                        logger.error(
                            f"❌ Failed to create collections '{collection_names}': {response.status}"
                        )
                        logger.error(f"Response JSON:\n{json.dumps(result, indent=2)}")
                        return False
        except Exception as e:
            logger.error(f"❌ Error creating collections '{collection_names}': {e}")
            return False

    async def _verify_collections(self) -> bool:
        """Verify collections are created and metadata schema is available"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.ingestor_server_url}/v1/collections"
                ) as response:
                    result = await response.json()
                    if response.status == 200:
                        collections = result.get("collections", [])
                        logger.info("✅ Collections retrieved successfully:")
                        logger.info(f"Response JSON:\n{json.dumps(result, indent=2)}")

                        # Check if our test collections exist
                        collection_names = [
                            col.get("collection_name") for col in collections
                        ]
                        expected_collections = self.expected_collections

                        missing_collections = [
                            name
                            for name in expected_collections
                            if name not in collection_names
                        ]
                        if missing_collections:
                            logger.error(
                                f"❌ Missing collections: {missing_collections}"
                            )
                            return False

                        # Verify metadata schema for collection with metadata
                        metadata_collection = self.collections["with_metadata"]
                        for collection in collections:
                            if collection.get("collection_name") == metadata_collection:
                                schema = collection.get("metadata_schema", [])
                                if not self._validate_metadata_schema(
                                    schema, metadata_collection
                                ):
                                    return False
                                break

                        # Verify metadata schema for custom metadata collection
                        custom_metadata_collection = self.CUSTOM_METADATA_COLLECTION
                        for collection in collections:
                            if (
                                collection.get("collection_name")
                                == custom_metadata_collection
                            ):
                                schema = collection.get("metadata_schema", [])
                                if not self._validate_custom_metadata_schema(
                                    schema, custom_metadata_collection
                                ):
                                    return False
                                break

                        logger.info(
                            f"✅ Collections verified successfully: {collection_names}"
                        )
                        return True
                    else:
                        logger.error(f"❌ Failed to get collections: {response.status}")
                        logger.error(f"Response JSON:\n{json.dumps(result, indent=2)}")
                        return False
        except Exception as e:
            logger.error(f"❌ Error verifying collections: {e}")
            return False

    def _validate_metadata_schema(
        self, actual_schema: list[dict[str, Any]], collection_name: str
    ) -> bool:
        """Validate that the actual metadata schema matches the expected schema"""
        expected_fields = {
            "timestamp": {
                "type": "datetime",
                "description": "Timestamp of when the document was created",
            },
            "meta_field_1": {
                "type": "string",
                "description": "Description for the document",
            },
        }

        actual_fields = {field["name"]: field for field in actual_schema}

        for field_name, expected_config in expected_fields.items():
            if field_name not in actual_fields:
                logger.error(
                    f"❌ Missing field '{field_name}' in collection '{collection_name}'"
                )
                return False

            actual_field = actual_fields[field_name]
            if actual_field.get("type") != expected_config["type"]:
                logger.error(
                    f"❌ Field '{field_name}' has wrong type. Expected: {expected_config['type']}, Got: {actual_field.get('type')}"
                )
                return False

            if actual_field.get("description") != expected_config["description"]:
                logger.error(
                    f"❌ Field '{field_name}' has wrong description. Expected: {expected_config['description']}, Got: {actual_field.get('description')}"
                )
                return False

        logger.info(
            f"✅ Metadata schema validation passed for collection '{collection_name}'"
        )
        return True

    def _validate_custom_metadata_schema(
        self, actual_schema: list[dict[str, Any]], collection_name: str
    ) -> bool:
        """Validate that the actual custom metadata schema matches the expected schema"""
        expected_fields = {
            "title": {
                "type": "string",
                "required": True,
                "max_length": 200,
                "description": "Document title",
            },
            "category": {
                "type": "string",
                "required": False,
                "description": "Document category",
            },
            "rating": {
                "type": "float",
                "required": False,
                "description": "Document rating",
            },
            "is_public": {
                "type": "boolean",
                "required": False,
                "description": "Whether document is public",
            },
            "tags": {
                "type": "array",
                "array_type": "string",
                "max_length": 50,
                "required": False,
                "description": "Document tags",
            },
            "created_date": {
                "type": "datetime",
                "required": True,
                "description": "Document creation date",
            },
            "updated_date": {
                "type": "datetime",
                "required": False,
                "description": "Document update date",
            },
        }

        actual_fields = {field["name"]: field for field in actual_schema}

        for field_name, expected_config in expected_fields.items():
            if field_name not in actual_fields:
                logger.error(
                    f"❌ Missing field '{field_name}' in collection '{collection_name}'"
                )
                return False

            actual_field = actual_fields[field_name]
            if actual_field.get("type") != expected_config["type"]:
                logger.error(
                    f"❌ Field '{field_name}' has wrong type. Expected: {expected_config['type']}, Got: {actual_field.get('type')}"
                )
                return False

            if actual_field.get("required") != expected_config["required"]:
                logger.error(
                    f"❌ Field '{field_name}' has wrong required flag. Expected: {expected_config['required']}, Got: {actual_field.get('required')}"
                )
                return False

            if (
                "max_length" in expected_config
                and actual_field.get("max_length") != expected_config["max_length"]
            ):
                logger.error(
                    f"❌ Field '{field_name}' has wrong max_length. Expected: {expected_config['max_length']}, Got: {actual_field.get('max_length')}"
                )
                return False

            if (
                "array_type" in expected_config
                and actual_field.get("array_type") != expected_config["array_type"]
            ):
                logger.error(
                    f"❌ Field '{field_name}' has wrong array_type. Expected: {expected_config['array_type']}, Got: {actual_field.get('array_type')}"
                )
                return False

            if actual_field.get("description") != expected_config["description"]:
                logger.error(
                    f"❌ Field '{field_name}' has wrong description. Expected: {expected_config['description']}, Got: {actual_field.get('description')}"
                )
                return False

        logger.info(
            f"✅ Custom metadata schema validation passed for collection '{collection_name}'"
        )
        return True

    @test_case(71, "Data Catalog Metadata")
    async def _test_data_catalog_metadata(self) -> bool:
        """Test data catalog metadata for collections and documents"""
        logger.info("\n=== Test 71: Data Catalog Metadata ===")
        test_start = time.time()

        try:
            # Test 1: Verify initial catalog metadata (created in Test 2)
            verify_initial_success = await self._verify_collection_catalog_metadata(
                self.CATALOG_COLLECTION,
                expected_description="Test catalog collection",
                expected_tags=["test", "integration"],
                expected_owner="Test Team",
            )
            if not verify_initial_success:
                test_time = time.time() - test_start
                self.add_test_result(
                    self._test_data_catalog_metadata.test_number,
                    self._test_data_catalog_metadata.test_name,
                    "Test data catalog metadata functionality including: 1) Verify initial catalog metadata from collection creation, 2) Update collection catalog metadata via PATCH, 3) Verify updates.",
                    [
                        "GET /v1/collections",
                        "PATCH /v1/collections/{collection_name}/metadata",
                    ],
                    [
                        "collection_name",
                        "description",
                        "tags",
                        "owner",
                        "created_by",
                        "business_domain",
                        "status",
                    ],
                    test_time,
                    TestStatus.FAILURE,
                    "Failed to verify initial collection catalog metadata",
                )
                return False

            # Test 2: Update collection catalog metadata
            update_success = await self._update_collection_catalog_metadata(
                self.CATALOG_COLLECTION,
                description="Updated description",
                tags=["updated", "production"],
                status="Archived",
            )
            if not update_success:
                test_time = time.time() - test_start
                self.add_test_result(
                    self._test_data_catalog_metadata.test_number,
                    self._test_data_catalog_metadata.test_name,
                    "Test data catalog metadata functionality including: 1) Verify initial catalog metadata from collection creation, 2) Update collection catalog metadata via PATCH, 3) Verify updates.",
                    [
                        "GET /v1/collections",
                        "PATCH /v1/collections/{collection_name}/metadata",
                    ],
                    [
                        "collection_name",
                        "description",
                        "tags",
                        "owner",
                        "created_by",
                        "business_domain",
                        "status",
                    ],
                    test_time,
                    TestStatus.FAILURE,
                    "Failed to update collection catalog metadata",
                )
                return False

            # Test 3: Verify updated catalog metadata
            verify_update_success = await self._verify_collection_catalog_metadata(
                self.CATALOG_COLLECTION,
                expected_description="Updated description",
                expected_tags=["updated", "production"],
                expected_status="Archived",
            )
            if not verify_update_success:
                test_time = time.time() - test_start
                self.add_test_result(
                    self._test_data_catalog_metadata.test_number,
                    self._test_data_catalog_metadata.test_name,
                    "Test data catalog metadata functionality including: 1) Verify initial catalog metadata from collection creation, 2) Update collection catalog metadata via PATCH, 3) Verify updates.",
                    [
                        "GET /v1/collections",
                        "PATCH /v1/collections/{collection_name}/metadata",
                    ],
                    [
                        "collection_name",
                        "description",
                        "tags",
                        "owner",
                        "created_by",
                        "business_domain",
                        "status",
                    ],
                    test_time,
                    TestStatus.FAILURE,
                    "Failed to verify updated collection catalog metadata",
                )
                return False

            # Test 4: Upload a test document to the catalog collection
            logger.info("Uploading test document for document metadata testing...")
            upload_success = await self._upload_test_document(self.CATALOG_COLLECTION)
            if not upload_success:
                test_time = time.time() - test_start
                self.add_test_result(
                    self._test_data_catalog_metadata.test_number,
                    self._test_data_catalog_metadata.test_name,
                    "Test data catalog metadata functionality including: 1) Verify initial catalog metadata from collection creation, 2) Update collection catalog metadata via PATCH, 3) Verify updates, 4) Upload test document, 5) Update document catalog metadata, 6) Verify document metadata.",
                    [
                        "GET /v1/collections",
                        "PATCH /v1/collections/{collection_name}/metadata",
                        "POST /v1/documents",
                        "PATCH /v1/collections/{collection_name}/documents/{document_name}/metadata",
                    ],
                    [
                        "collection_name",
                        "description",
                        "tags",
                        "document_name",
                    ],
                    test_time,
                    TestStatus.FAILURE,
                    "Failed to upload test document",
                )
                return False

            # Test 4.5: Verify catalog metadata was applied during upload
            test_document_name = "test_catalog_doc.txt"
            doc_verify_upload_success = await self._verify_document_catalog_metadata(
                self.CATALOG_COLLECTION,
                test_document_name,
                expected_description="Test document uploaded with catalog metadata",
                expected_tags=["test", "upload", "catalog"],
            )
            if not doc_verify_upload_success:
                test_time = time.time() - test_start
                self.add_test_result(
                    self._test_data_catalog_metadata.test_number,
                    self._test_data_catalog_metadata.test_name,
                    "Test data catalog metadata functionality including: 1) Verify initial catalog metadata from collection creation, 2) Update collection catalog metadata via PATCH, 3) Verify updates, 4) Upload test document WITH catalog metadata, 4.5) Verify upload catalog metadata, 5) Update document catalog metadata, 6) Verify document metadata.",
                    [
                        "GET /v1/collections",
                        "PATCH /v1/collections/{collection_name}/metadata",
                        "POST /v1/documents",
                        "GET /v1/documents",
                        "PATCH /v1/collections/{collection_name}/documents/{document_name}/metadata",
                    ],
                    [
                        "collection_name",
                        "description",
                        "tags",
                        "document_name",
                    ],
                    test_time,
                    TestStatus.FAILURE,
                    "Failed to verify catalog metadata set during document upload",
                )
                return False

            # Test 5: Update document catalog metadata
            doc_update_success = await self._update_document_catalog_metadata(
                self.CATALOG_COLLECTION,
                test_document_name,
                description="Test document description",
                tags=["doc-tag", "test"],
            )
            if not doc_update_success:
                test_time = time.time() - test_start
                self.add_test_result(
                    self._test_data_catalog_metadata.test_number,
                    self._test_data_catalog_metadata.test_name,
                    "Test data catalog metadata functionality including: 1) Verify initial catalog metadata from collection creation, 2) Update collection catalog metadata via PATCH, 3) Verify updates, 4) Upload test document, 5) Update document catalog metadata, 6) Verify document metadata.",
                    [
                        "GET /v1/collections",
                        "PATCH /v1/collections/{collection_name}/metadata",
                        "POST /v1/documents",
                        "PATCH /v1/collections/{collection_name}/documents/{document_name}/metadata",
                    ],
                    [
                        "collection_name",
                        "description",
                        "tags",
                        "document_name",
                    ],
                    test_time,
                    TestStatus.FAILURE,
                    "Failed to update document catalog metadata",
                )
                return False

            # Test 6: Verify document catalog metadata
            doc_verify_success = await self._verify_document_catalog_metadata(
                self.CATALOG_COLLECTION,
                test_document_name,
                expected_description="Test document description",
                expected_tags=["doc-tag", "test"],
            )
            if not doc_verify_success:
                test_time = time.time() - test_start
                self.add_test_result(
                    self._test_data_catalog_metadata.test_number,
                    self._test_data_catalog_metadata.test_name,
                    "Test data catalog metadata functionality including: 1) Verify initial catalog metadata from collection creation, 2) Update collection catalog metadata via PATCH, 3) Verify updates, 4) Upload test document, 5) Update document catalog metadata, 6) Verify document metadata.",
                    [
                        "GET /v1/collections",
                        "PATCH /v1/collections/{collection_name}/metadata",
                        "POST /v1/documents",
                        "PATCH /v1/collections/{collection_name}/documents/{document_name}/metadata",
                    ],
                    [
                        "collection_name",
                        "description",
                        "tags",
                        "document_name",
                    ],
                    test_time,
                    TestStatus.FAILURE,
                    "Failed to verify document catalog metadata",
                )
                return False

            test_time = time.time() - test_start
            self.add_test_result(
                self._test_data_catalog_metadata.test_number,
                self._test_data_catalog_metadata.test_name,
                "Test data catalog metadata functionality including: 1) Verify initial catalog metadata from collection creation, 2) Update collection catalog metadata via PATCH, 3) Verify updates, 4) Upload test document, 5) Update document catalog metadata, 6) Verify document metadata. Tests both collection and document level catalog metadata endpoints.",
                [
                    "GET /v1/collections",
                    "PATCH /v1/collections/{collection_name}/metadata",
                    "POST /v1/documents",
                    "PATCH /v1/collections/{collection_name}/documents/{document_name}/metadata",
                    "GET /v1/documents",
                ],
                [
                    "collection_name",
                    "description",
                    "tags",
                    "document_name",
                ],
                test_time,
                TestStatus.SUCCESS,
            )
            return True
        finally:
            # Clean up the catalog test collection
            logger.info("Cleaning up catalog test collection...")
            await self._cleanup_catalog_collection(self.CATALOG_COLLECTION)

    async def _create_collection_with_catalog(
        self,
        collection_name: str,
        description: str = "",
        tags: list[str] | None = None,
        owner: str = "",
        created_by: str = "",
        business_domain: str = "",
        status: str = "Active",
    ) -> bool:
        """Create a collection with catalog metadata"""
        try:
            payload = {
                "collection_name": collection_name,
                "embedding_dimension": 2048,
                "description": description,
                "tags": tags or [],
                "owner": owner,
                "created_by": created_by,
                "business_domain": business_domain,
                "status": status,
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ingestor_server_url}/v1/collection", json=payload
                ) as response:
                    result = await response.json()
                    if response.status == 200:
                        logger.info(
                            f"✅ Collection '{collection_name}' created with catalog metadata:"
                        )
                        logger.info(f"Response JSON:\n{json.dumps(result, indent=2)}")
                        return True
                    else:
                        logger.error(
                            f"❌ Failed to create collection '{collection_name}' with catalog metadata: {response.status}"
                        )
                        logger.error(f"Response JSON:\n{json.dumps(result, indent=2)}")
                        return False
        except Exception:
            logger.exception(
                "❌ Error creating collection '%s' with catalog metadata",
                collection_name,
            )
            return False

    async def _verify_collection_catalog_metadata(
        self,
        collection_name: str,
        expected_description: str | None = None,
        expected_tags: list[str] | None = None,
        expected_owner: str | None = None,
        expected_status: str | None = None,
    ) -> bool:
        """Verify collection catalog metadata"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.ingestor_server_url}/v1/collections"
                ) as response:
                    result = await response.json()
                    if response.status == 200:
                        collections = result.get("collections", [])
                        for collection in collections:
                            if collection.get("collection_name") == collection_name:
                                # Get collection_info which contains catalog metadata
                                collection_info = collection.get("collection_info", {})

                                # Check catalog metadata fields
                                if expected_description is not None:
                                    actual_description = collection_info.get(
                                        "description", ""
                                    )
                                    if actual_description != expected_description:
                                        logger.error(
                                            f"❌ Description mismatch. Expected: '{expected_description}', Got: '{actual_description}'"
                                        )
                                        return False

                                if expected_tags is not None:
                                    actual_tags = collection_info.get("tags", [])
                                    if set(actual_tags) != set(expected_tags):
                                        logger.error(
                                            f"❌ Tags mismatch. Expected: {expected_tags}, Got: {actual_tags}"
                                        )
                                        return False

                                if expected_owner is not None:
                                    actual_owner = collection_info.get("owner", "")
                                    if actual_owner != expected_owner:
                                        logger.error(
                                            f"❌ Owner mismatch. Expected: '{expected_owner}', Got: '{actual_owner}'"
                                        )
                                        return False

                                if expected_status is not None:
                                    actual_status = collection_info.get("status", "")
                                    if actual_status != expected_status:
                                        logger.error(
                                            f"❌ Status mismatch. Expected: '{expected_status}', Got: '{actual_status}'"
                                        )
                                        return False

                                # Verify required fields exist
                                if "date_created" not in collection_info:
                                    logger.error("❌ Missing 'date_created' field")
                                    return False

                                if "last_updated" not in collection_info:
                                    logger.error("❌ Missing 'last_updated' field")
                                    return False

                                logger.info(
                                    f"✅ Catalog metadata verification passed for collection '{collection_name}'"
                                )
                                logger.info(
                                    f"Collection data:\n{json.dumps(collection, indent=2)}"
                                )
                                return True

                        logger.error(f"❌ Collection '{collection_name}' not found")
                        return False
                    else:
                        logger.error(f"❌ Failed to get collections: {response.status}")
                        return False
        except Exception:
            logger.exception(
                "❌ Error verifying catalog metadata for collection '%s'",
                collection_name,
            )
            return False

    async def _update_collection_catalog_metadata(
        self,
        collection_name: str,
        description: str | None = None,
        tags: list[str] | None = None,
        owner: str | None = None,
        status: str | None = None,
    ) -> bool:
        """Update collection catalog metadata"""
        try:
            payload = {}
            if description is not None:
                payload["description"] = description
            if tags is not None:
                payload["tags"] = tags
            if owner is not None:
                payload["owner"] = owner
            if status is not None:
                payload["status"] = status

            async with aiohttp.ClientSession() as session:
                async with session.patch(
                    f"{self.ingestor_server_url}/v1/collections/{collection_name}/metadata",
                    json=payload,
                ) as response:
                    result = await response.json()
                    if response.status == 200:
                        logger.info(
                            f"✅ Catalog metadata updated for collection '{collection_name}':"
                        )
                        logger.info(f"Response JSON:\n{json.dumps(result, indent=2)}")
                        return True
                    else:
                        logger.error(
                            f"❌ Failed to update catalog metadata for '{collection_name}': {response.status}"
                        )
                        logger.error(f"Response JSON:\n{json.dumps(result, indent=2)}")
                        return False
        except Exception:
            logger.exception(
                "❌ Error updating catalog metadata for collection '%s'",
                collection_name,
            )
            return False

    async def _upload_test_document(self, collection_name: str) -> bool:
        """Upload a simple test document to the collection"""
        try:
            # Create a simple text file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, prefix="test_catalog_doc_"
            ) as f:
                f.write("This is a test document for catalog metadata testing.")
                temp_file_path = f.name

            try:
                # Upload the document using correct multipart format
                async with aiohttp.ClientSession() as session:
                    # Read file content asynchronously
                    file_content = await asyncio.to_thread(
                        Path(temp_file_path).read_bytes
                    )

                    # Prepare form data matching the API schema
                    form_data = aiohttp.FormData()
                    form_data.add_field(
                        "documents",
                        file_content,
                        filename="test_catalog_doc.txt",
                        content_type="text/plain",
                    )

                    # Add JSON data field WITH documents_catalog_metadata
                    upload_data = {
                        "collection_name": collection_name,
                        "blocking": True,
                        "split_options": {"chunk_size": 512, "chunk_overlap": 150},
                        "custom_metadata": [],
                        "generate_summary": False,
                        "documents_catalog_metadata": [
                            {
                                "filename": "test_catalog_doc.txt",
                                "description": "Test document uploaded with catalog metadata",
                                "tags": ["test", "upload", "catalog"],
                            }
                        ],
                    }
                    form_data.add_field(
                        "data",
                        json.dumps(upload_data),
                        content_type="application/json",
                    )

                    async with session.post(
                        f"{self.ingestor_server_url}/v1/documents", data=form_data
                    ) as response:
                        result = await response.json()
                        if response.status == 200:
                            logger.info(
                                "✅ Test document uploaded successfully with catalog metadata"
                            )
                            return True
                        else:
                            logger.error(
                                f"❌ Failed to upload test document: {response.status}"
                            )
                            logger.error(
                                f"Response JSON:\n{json.dumps(result, indent=2)}"
                            )
                            return False
            finally:
                # Clean up temp file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)

        except Exception:
            logger.exception(
                "❌ Error uploading test document to collection '%s'", collection_name
            )
            return False

    async def _update_document_catalog_metadata(
        self,
        collection_name: str,
        document_name: str,
        description: str | None = None,
        tags: list[str] | None = None,
    ) -> bool:
        """Update document catalog metadata"""
        try:
            payload = {}
            if description is not None:
                payload["description"] = description
            if tags is not None:
                payload["tags"] = tags

            async with aiohttp.ClientSession() as session:
                async with session.patch(
                    f"{self.ingestor_server_url}/v1/collections/{collection_name}/documents/{document_name}/metadata",
                    json=payload,
                ) as response:
                    result = await response.json()
                    if response.status == 200:
                        logger.info(
                            f"✅ Document catalog metadata updated for '{document_name}'"
                        )
                        logger.info(f"Response JSON:\n{json.dumps(result, indent=2)}")
                        return True
                    else:
                        logger.error(
                            f"❌ Failed to update document catalog metadata: {response.status}"
                        )
                        logger.error(f"Response JSON:\n{json.dumps(result, indent=2)}")
                        return False
        except Exception:
            logger.exception(
                "❌ Error updating document catalog metadata for '%s' in collection '%s'",
                document_name,
                collection_name,
            )
            return False

    async def _verify_document_catalog_metadata(
        self,
        collection_name: str,
        document_name: str,
        expected_description: str | None = None,
        expected_tags: list[str] | None = None,
    ) -> bool:
        """Verify document catalog metadata"""
        try:
            async with aiohttp.ClientSession() as session:
                params = {"collection_name": collection_name}
                async with session.get(
                    f"{self.ingestor_server_url}/v1/documents", params=params
                ) as response:
                    result = await response.json()
                    if response.status == 200:
                        documents = result.get("documents", [])
                        for document in documents:
                            if document.get("document_name") == document_name:
                                # Catalog metadata is in document_info, not metadata
                                document_info = document.get("document_info", {})

                                if expected_description is not None:
                                    actual_description = document_info.get(
                                        "description", ""
                                    )
                                    if actual_description != expected_description:
                                        logger.error(
                                            f"❌ Document description mismatch. Expected: '{expected_description}', Got: '{actual_description}'"
                                        )
                                        logger.error(
                                            f"Document structure:\n{json.dumps(document, indent=2)}"
                                        )
                                        return False

                                if expected_tags is not None:
                                    actual_tags = document_info.get("tags", [])
                                    if set(actual_tags) != set(expected_tags):
                                        logger.error(
                                            f"❌ Document tags mismatch. Expected: {expected_tags}, Got: {actual_tags}"
                                        )
                                        return False

                                logger.info(
                                    f"✅ Document catalog metadata verification passed for '{document_name}'"
                                )
                                logger.info(
                                    f"Document info:\n{json.dumps(document_info, indent=2)}"
                                )
                                return True

                        logger.error(f"❌ Document '{document_name}' not found")
                        return False
                    else:
                        logger.error(f"❌ Failed to get documents: {response.status}")
                        return False
        except Exception:
            logger.exception(
                "❌ Error verifying document catalog metadata for '%s' in collection '%s'",
                document_name,
                collection_name,
            )
            return False

    async def _cleanup_catalog_collection(self, collection_name: str) -> bool:
        """Clean up catalog test collection"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.delete(
                    f"{self.ingestor_server_url}/v1/collections",
                    json=[collection_name],
                ) as response:
                    result = await response.json()
                    if response.status == 200:
                        logger.info(
                            f"✅ Catalog test collection '{collection_name}' cleaned up successfully"
                        )
                        return True
                    else:
                        logger.error(
                            f"❌ Failed to clean up collection '{collection_name}': {response.status}"
                        )
                        logger.error(f"Response JSON:\n{json.dumps(result, indent=2)}")
                        return False
        except Exception as e:
            logger.error(f"❌ Error cleaning up collection: {e}")
            return False

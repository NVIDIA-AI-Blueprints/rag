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
Cleanup test module
"""

import json
import logging
import os
import time

import aiohttp

from ..base import BaseTestModule, TestStatus, test_case
from ..utils.file_utils import get_test_files

logger = logging.getLogger(__name__)


class CleanupModule(BaseTestModule):
    """Cleanup test module"""

    # Additional collections that need to be cleaned up
    CUSTOM_METADATA_COLLECTION = "test_custom_metadata_collection"
    TEST_FILE_TYPES_COLLECTION = "test_file_types"
    CSV_DELETION_BATCH_COLLECTION = "test_csv_deletion_batch"
    CATALOG_COLLECTION = "test_catalog_collection"

    @test_case(16, "Delete Documents")
    async def _test_delete_documents(self) -> bool:
        """Test deleting documents"""
        logger.info("\n=== Test 16: Delete Documents ===")
        delete_docs_start = time.time()
        file_names_with_metadata = [
            os.path.basename(f) for f in self.test_runner.test_files
        ]
        # Get files for collection without metadata
        test_files_without_metadata = self._get_test_files(3, "without_metadata")
        file_names_without_metadata = [
            os.path.basename(f) for f in test_files_without_metadata
        ]
        delete_with_metadata = await self._delete_documents(
            self.collections["with_metadata"], file_names_with_metadata
        )
        delete_without_metadata = await self._delete_documents(
            self.collections["without_metadata"], file_names_without_metadata
        )
        delete_docs_time = time.time() - delete_docs_start

        if delete_with_metadata and delete_without_metadata:
            self.add_test_result(
                self._test_delete_documents.test_number,
                self._test_delete_documents.test_name,
                f"Delete all test documents from both collections using DELETE endpoint. Collections: {self.collections['with_metadata']} (files: {', '.join(file_names_with_metadata)}), {self.collections['without_metadata']} (files: {', '.join(file_names_without_metadata)}). Supports bulk deletion of multiple files and verification of deletion success.",
                ["DELETE /v1/documents"],
                ["collection_name", "file_names"],
                delete_docs_time,
                TestStatus.SUCCESS,
            )
            return True
        else:
            self.add_test_result(
                self._test_delete_documents.test_number,
                self._test_delete_documents.test_name,
                f"Delete all test documents from both collections using DELETE endpoint. Collections: {self.collections['with_metadata']} (files: {', '.join(file_names_with_metadata)}), {self.collections['without_metadata']} (files: {', '.join(file_names_without_metadata)}). Supports bulk deletion of multiple files and verification of deletion success.",
                ["DELETE /v1/documents"],
                ["collection_name", "file_names"],
                delete_docs_time,
                TestStatus.FAILURE,
                "Failed to delete documents from one or both collections",
            )
            return False

    @test_case(17, "Verify Document Deletion")
    async def _test_verify_document_deletion(self) -> bool:
        """Test verifying document deletion"""
        logger.info("\n=== Test 17: Verify Document Deletion ===")
        verify_delete_start = time.time()
        verify_delete_with_metadata = await self._verify_documents(
            self.collections["with_metadata"], None, []
        )
        verify_delete_without_metadata = await self._verify_documents(
            self.collections["without_metadata"], None, []
        )
        verify_delete_time = time.time() - verify_delete_start

        if verify_delete_with_metadata and verify_delete_without_metadata:
            self.add_test_result(
                self._test_verify_document_deletion.test_number,
                self._test_verify_document_deletion.test_name,
                f"Verify that all documents have been successfully deleted from collections. Collections: {self.collections['with_metadata']}, {self.collections['without_metadata']}. Validates that collections are empty after deletion and confirms complete cleanup.",
                ["GET /v1/documents"],
                ["collection_name", "documents"],
                verify_delete_time,
                TestStatus.SUCCESS,
            )
            return True
        else:
            self.add_test_result(
                self._test_verify_document_deletion.test_number,
                self._test_verify_document_deletion.test_name,
                f"Verify that all documents have been successfully deleted from collections. Collections: {self.collections['with_metadata']}, {self.collections['without_metadata']}. Validates that collections are empty after deletion and confirms complete cleanup.",
                ["GET /v1/documents"],
                ["collection_name", "documents"],
                verify_delete_time,
                TestStatus.FAILURE,
                "Document deletion verification failed",
            )
            return False

    @test_case(18, "Delete Collections")
    async def _test_delete_collections(self) -> bool:
        """Test deleting collections"""
        logger.info("\n=== Test 18: Delete Collections ===")
        delete_collections_start = time.time()

        # Include all collections: base collections + additional collections created in test 2
        # Use a set to avoid duplicates in case some collections are already in base collections
        all_collections = list(
            set(
                list(self.collections.values())
                + [
                    self.CUSTOM_METADATA_COLLECTION,
                    self.TEST_FILE_TYPES_COLLECTION,
                    self.CSV_DELETION_BATCH_COLLECTION,
                    self.CATALOG_COLLECTION,
                ]
            )
        )

        collections_deleted = await self._delete_collections(all_collections)
        delete_collections_time = time.time() - delete_collections_start

        if collections_deleted:
            self.add_test_result(
                self._test_delete_collections.test_number,
                self._test_delete_collections.test_name,
                f"Delete all test collections to clean up the test environment. Collections: {', '.join(all_collections)}.",
                ["DELETE /v1/collections"],
                ["collection_names"],
                delete_collections_time,
                TestStatus.SUCCESS,
            )
            return True
        else:
            self.add_test_result(
                self._test_delete_collections.test_number,
                self._test_delete_collections.test_name,
                f"Delete all test collections to clean up the test environment. Collections: {', '.join(all_collections)}.",
                ["DELETE /v1/collections"],
                ["collection_names"],
                delete_collections_time,
                TestStatus.FAILURE,
                "Failed to delete collections",
            )
            return False

    @test_case(19, "Verify Collection Deletion")
    async def _test_verify_collection_deletion(self) -> bool:
        """Test verifying collection deletion"""
        logger.info("\n=== Test 19: Verify Collection Deletion ===")
        verify_collections_delete_start = time.time()
        collections_verified = await self._verify_collections_deleted()
        verify_collections_delete_time = time.time() - verify_collections_delete_start

        # Include all collections for verification message
        # Use a set to avoid duplicates in case some collections are already in base collections
        all_collections = list(
            set(
                list(self.collections.values())
                + [
                    self.CUSTOM_METADATA_COLLECTION,
                    self.TEST_FILE_TYPES_COLLECTION,
                    self.CSV_DELETION_BATCH_COLLECTION,
                    self.CATALOG_COLLECTION,
                ]
            )
        )

        if collections_verified:
            self.add_test_result(
                self._test_verify_collection_deletion.test_number,
                self._test_verify_collection_deletion.test_name,
                f"Verify that all test collections have been successfully deleted. Collections: {', '.join(all_collections)}.",
                ["GET /v1/collections"],
                ["collections[].collection_name"],
                verify_collections_delete_time,
                TestStatus.SUCCESS,
            )
            return True
        else:
            self.add_test_result(
                self._test_verify_collection_deletion.test_number,
                self._test_verify_collection_deletion.test_name,
                f"Verify that all test collections have been successfully deleted. Collections: {', '.join(all_collections)}.",
                ["GET /v1/collections"],
                ["collections[].collection_name"],
                verify_collections_delete_time,
                TestStatus.FAILURE,
                "Collection deletion verification failed",
            )
            return False

    def _get_test_files(
        self, count: int = 3, collection_type: str = "with_metadata"
    ) -> list[str]:
        """Get test files from the data directory or specified files"""
        return get_test_files(
            self.test_runner.data_dir,
            count,
            collection_type,
            self.test_runner.files_with_metadata,
            self.test_runner.files_without_metadata,
        )

    async def _delete_documents(
        self, collection_name: str, file_names: list[str]
    ) -> bool:
        """Delete documents from a collection"""
        async with aiohttp.ClientSession() as session:
            try:
                # DELETE /v1/documents expects query params:
                # collection_name=<name>&document_names=<file1>&document_names=<file2>...
                params = [("collection_name", collection_name)]
                for name in file_names:
                    params.append(("document_names", name))
                logger.info(
                    f"🗑️ Deleting documents from collection '{collection_name}':"
                )
                logger.info(f"📋 Delete request params: {params}")
                logger.info(f"📁 Files to delete: {json.dumps(file_names, indent=2)}")

                async with session.delete(
                    f"{self.ingestor_server_url}/v1/documents",
                    params=params
                ) as response:
                    result = await response.json()
                    if response.status == 200:
                        logger.info(
                            f"✅ Documents deleted successfully from '{collection_name}':"
                        )
                        logger.info(f"Response JSON:\n{json.dumps(result, indent=2)}")
                        return True
                    else:
                        logger.error(
                            f"❌ Failed to delete documents from '{collection_name}': {response.status}"
                        )
                        logger.error(f"Response JSON:\n{json.dumps(result, indent=2)}")
                        return False
            except Exception as e:
                logger.error(f"❌ Error deleting documents: {e}")
                return False

    async def _delete_collections(self, collection_names: list[str]) -> bool:
        """Delete collections"""
        async with aiohttp.ClientSession() as session:
            try:
                logger.info("🗑️ Deleting collections:")
                logger.info(
                    f"📋 Collections to delete: {json.dumps(collection_names, indent=2)}"
                )

                # DELETE /v1/collections expects query params:
                # collection_names=<col1>&collection_names=<col2>...
                params = [("collection_names", name) for name in collection_names]
                async with session.delete(f"{self.ingestor_server_url}/v1/collections", params=params) as response:
                    result = await response.json()
                    if response.status == 200:
                        logger.info("✅ Collections deleted successfully:")
                        logger.info(f"Response JSON:\n{json.dumps(result, indent=2)}")
                        return True
                    else:
                        logger.error(
                            f"❌ Failed to delete collections: {response.status}"
                        )
                        logger.error(f"Response JSON:\n{json.dumps(result, indent=2)}")
                        return False
            except Exception as e:
                logger.error(f"❌ Error deleting collections: {e}")
                return False

    async def _verify_collections_deleted(self) -> bool:
        """Verify that test collections have been successfully deleted"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.ingestor_server_url}/v1/collections"
                ) as response:
                    result = await response.json()
                    if response.status == 200:
                        logger.info("✅ Collections verification successful:")
                        logger.info(f"Response JSON:\n{json.dumps(result, indent=2)}")

                        collections = result.get("collections", [])
                        collection_names = [
                            col.get("collection_name") for col in collections
                        ]

                        # Check if any of our test collections still exist
                        # Include all collections that should have been deleted
                        # Use a set to avoid duplicates in case some collections are already in base collections
                        all_test_collections = list(
                            set(
                                list(self.collections.values())
                                + [
                                    self.CUSTOM_METADATA_COLLECTION,
                                    self.TEST_FILE_TYPES_COLLECTION,
                                    self.CATALOG_COLLECTION,
                                ]
                            )
                        )

                        remaining_test_collections = []
                        for collection_name in all_test_collections:
                            if collection_name in collection_names:
                                remaining_test_collections.append(collection_name)

                        if remaining_test_collections:
                            logger.error(
                                f"❌ Test collections still exist: {remaining_test_collections}"
                            )
                            logger.error(f"All collections: {collection_names}")
                            return False
                        else:
                            logger.info("✅ All test collections successfully deleted")
                            return True
                    else:
                        logger.error(
                            f"❌ Failed to get collections for verification: {response.status}"
                        )
                        logger.error(f"Response JSON:\n{json.dumps(result, indent=2)}")
                        return False
        except Exception as e:
            logger.error(f"❌ Error verifying collection deletion: {e}")
            return False

    async def _verify_documents(
        self,
        collection_name: str,
        expected_metadata=None,
        expected_documents=None,
    ) -> bool:
        """Verify documents are ingested in the collection and optionally verify metadata and document list"""
        async with aiohttp.ClientSession() as session:
            try:
                params = {"collection_name": collection_name}
                async with session.get(
                    f"{self.ingestor_server_url}/v1/documents", params=params
                ) as response:
                    result = await response.json()
                    if response.status == 200:
                        documents = result.get("documents", [])
                        logger.info(
                            f"Found {len(documents)} documents in collection '{collection_name}'"
                        )

                        # Extract actual document names from the collection
                        actual_document_names = [
                            doc.get("document_name") for doc in documents
                        ]
                        logger.info(
                            f"Actual documents in collection: {actual_document_names}"
                        )

                        # Verify that only expected documents are present (if expected_documents provided)
                        if expected_documents is not None:
                            expected_doc_set = set(expected_documents)
                            actual_doc_set = set(actual_document_names)

                            # Check for unexpected documents
                            unexpected_documents = actual_doc_set - expected_doc_set
                            if unexpected_documents:
                                logger.error(
                                    f"❌ Unexpected documents found in collection '{collection_name}': {unexpected_documents}"
                                )
                                logger.error(
                                    f"Expected documents: {expected_documents}"
                                )
                                logger.error(
                                    f"Actual documents: {actual_document_names}"
                                )
                                return False

                            # Check for missing documents
                            missing_documents = expected_doc_set - actual_doc_set
                            if missing_documents:
                                logger.error(
                                    f"❌ Missing documents in collection '{collection_name}': {missing_documents}"
                                )
                                logger.error(
                                    f"Expected documents: {expected_documents}"
                                )
                                logger.error(
                                    f"Actual documents: {actual_document_names}"
                                )
                                return False

                            logger.info(
                                f"✅ Document list verification passed for collection '{collection_name}'"
                            )

                        # For deletion verification, we want to return True when documents are empty
                        # For normal verification, we want to return True when documents exist
                        if (
                            expected_documents is not None
                            and len(expected_documents) == 0
                        ):
                            # This is a deletion verification - return True if no documents found
                            return len(documents) == 0
                        else:
                            # This is a normal verification - return True if documents exist
                            return len(documents) > 0
                    else:
                        logger.error("❌ Failed to verify documents")
                        return False
            except Exception as e:
                logger.error(f"❌ Error verifying documents: {e}")
                return False

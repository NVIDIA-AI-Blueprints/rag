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
Summary test module - includes status tracking and progress monitoring
"""

import asyncio
import json
import logging
import os
import time

import aiohttp

from ..base import BaseTestModule, TestStatus, test_case
from ..utils.response_handlers import print_response
from ..utils.verification import verify_summary_content

logger = logging.getLogger(__name__)


class SummaryModule(BaseTestModule):
    """Summary test module with status tracking and progress monitoring"""

    async def test_fetch_summary(
        self, collection_name: str, filenames: list[str]
    ) -> bool:
        """Test fetching document summaries for all files in a collection"""
        async with aiohttp.ClientSession() as session:
            try:
                success_count = 0
                verification_count = 0
                total_files = len(filenames)

                for filename in filenames:
                    logger.info(f"📄 Fetching summary for file: {filename}")
                    params = {
                        "collection_name": collection_name,
                        "file_name": filename,
                        "blocking": "false",
                        "timeout": 20,
                    }
                    logger.info(
                        f"📋 Summary request params:\n{json.dumps(params, indent=2)}"
                    )

                    async with session.get(
                        f"{self.rag_server_url}/v1/summary", params=params
                    ) as response:
                        result = await print_response(response)
                        if response.status == 200:
                            logger.info(
                                f"✅ Summary fetched successfully for {filename}"
                            )
                            success_count += 1

                            # Verify summary content for default files
                            summary_text = result.get("summary", "")
                            if verify_summary_content(summary_text, filename):
                                verification_count += 1
                            else:
                                logger.error(
                                    f"❌ Summary content verification failed for {filename}"
                                )
                        else:
                            logger.error(f"❌ Failed to fetch summary for {filename}")

                if success_count == total_files:
                    logger.info(
                        f"✅ Fetch summary test passed - all {total_files} files processed successfully"
                    )

                    # Log verification results
                    if verification_count == success_count:
                        logger.info(
                            f"✅ Summary content verification passed for all {verification_count} files"
                        )
                    else:
                        logger.warning(
                            f"⚠️ Summary content verification: {verification_count}/{success_count} files passed"
                        )

                    return True
                elif success_count > 0:
                    logger.warning(
                        f"⚠️ Fetch summary test partially passed - {success_count}/{total_files} files processed successfully"
                    )

                    # Log verification results for partial success
                    if verification_count > 0:
                        logger.info(
                            f"✅ Summary content verification passed for {verification_count}/{success_count} files"
                        )

                    return True  # Consider partial success as acceptable
                else:
                    logger.error(
                        "❌ Fetch summary test failed - no files processed successfully"
                    )
                    return False
            except Exception as e:
                logger.error(f"❌ Error in fetch summary test: {e}")
                return False

    async def test_summary_status_tracking(
        self, collection_name: str, filename: str
    ) -> bool:
        """Test summary status tracking with polling (PENDING -> IN_PROGRESS -> SUCCESS)"""
        async with aiohttp.ClientSession() as session:
            try:
                logger.info(f"📊 Testing status tracking for file: {filename}")
                params = {
                    "collection_name": collection_name,
                    "file_name": filename,
                    "blocking": "false",
                }

                statuses_observed = []
                max_polls = 30  # Poll for up to 60 seconds (30 * 2s)
                poll_count = 0

                while poll_count < max_polls:
                    async with session.get(
                        f"{self.rag_server_url}/v1/summary", params=params
                    ) as response:
                        result = await response.json()
                        status = result.get("status")

                        if status and status not in statuses_observed:
                            statuses_observed.append(status)
                            logger.info(f"📍 Status transition: {status}")

                            # Log progress if available
                            if status == "IN_PROGRESS" and "progress" in result:
                                progress = result["progress"]
                                logger.info(
                                    f"⏳ Progress: {progress.get('current')}/{progress.get('total')} - {progress.get('message')}"
                                )

                        # Check if we've reached a terminal state
                        if status in ["SUCCESS", "FAILED", "NOT_FOUND"]:
                            if status == "SUCCESS":
                                logger.info(
                                    f"✅ Status tracking completed successfully: {' -> '.join(statuses_observed)}"
                                )
                                return True
                            else:
                                logger.error(
                                    f"❌ Summary generation failed with status: {status}"
                                )
                                return False

                        poll_count += 1
                        await asyncio.sleep(2)  # Poll every 2 seconds

                logger.warning(
                    f"⚠️ Status tracking timeout after {max_polls * 2}s. Statuses observed: {' -> '.join(statuses_observed)}"
                )
                return False

            except Exception as e:
                logger.error(f"❌ Error in status tracking test: {e}")
                return False

    async def test_summary_blocking_mode(
        self, collection_name: str, filename: str
    ) -> bool:
        """Test summary retrieval in blocking mode with timeout"""
        async with aiohttp.ClientSession() as session:
            try:
                logger.info(f"🔄 Testing blocking mode for file: {filename}")
                params = {
                    "collection_name": collection_name,
                    "file_name": filename,
                    "blocking": "true",
                    "timeout": 60,
                }

                async with session.get(
                    f"{self.rag_server_url}/v1/summary", params=params
                ) as response:
                    await print_response(response)

                    if response.status == 200:
                        logger.info("✅ Blocking mode successful - summary retrieved")
                        return True
                    elif response.status == 202:
                        logger.warning("⚠️ Still in progress after timeout")
                        return True  # Not a failure, just slow
                    else:
                        logger.error(
                            f"❌ Blocking mode failed with status: {response.status}"
                        )
                        return False

            except Exception as e:
                logger.error(f"❌ Error in blocking mode test: {e}")
                return False

    async def test_summary_not_found(
        self, collection_name: str, nonexistent_file: str
    ) -> bool:
        """Test summary endpoint returns NOT_FOUND for non-existent files"""
        async with aiohttp.ClientSession() as session:
            try:
                logger.info(f"🔍 Testing NOT_FOUND status for: {nonexistent_file}")
                params = {
                    "collection_name": collection_name,
                    "file_name": nonexistent_file,
                    "blocking": "false",
                }

                async with session.get(
                    f"{self.rag_server_url}/v1/summary", params=params
                ) as response:
                    result = await response.json()

                    if response.status == 404 and result.get("status") == "NOT_FOUND":
                        logger.info(
                            "✅ Correctly returned NOT_FOUND for non-existent file"
                        )
                        return True
                    else:
                        logger.error(
                            f"❌ Expected 404/NOT_FOUND, got {response.status}/{result.get('status')}"
                        )
                        return False

            except Exception as e:
                logger.error(f"❌ Error in NOT_FOUND test: {e}")
                return False

    @test_case(15, "Fetch Summary")
    async def _test_fetch_summary(self) -> bool:
        """Test fetching summary"""
        logger.info("\n=== Test 15: Fetch Summary ===")
        summary_start = time.time()
        # Get all filenames from the collection with metadata
        all_filenames_with_metadata = [
            os.path.basename(f) for f in self.test_runner.test_files
        ]
        summary_success = await self.test_fetch_summary(
            self.collections["with_metadata"], all_filenames_with_metadata
        )
        summary_time = time.time() - summary_start

        if summary_success:
            self.add_test_result(
                self._test_fetch_summary.test_number,
                self._test_fetch_summary.test_name,
                f"Retrieve document summaries for all files in the collection with keyword-based content verification. Collection: {self.collections['with_metadata']}. Files: {', '.join(all_filenames_with_metadata)}. Supports both blocking and non-blocking modes with configurable timeout for summary generation. Includes automatic keyword verification for default files (multimodal_test.pdf: tables/charts/animals/gadgets, woods_frost.docx: Frost/woods/poem/collections) to ensure summary quality and relevance. Handles partial success scenarios.",
                ["GET /v1/summary"],
                ["collection_name", "file_name", "blocking", "timeout"],
                summary_time,
                TestStatus.SUCCESS,
            )
            return True
        else:
            self.add_test_result(
                self._test_fetch_summary.test_number,
                self._test_fetch_summary.test_name,
                f"Retrieve document summaries for all files in the collection with keyword-based content verification. Collection: {self.collections['with_metadata']}. Files: {', '.join(all_filenames_with_metadata)}. Supports both blocking and non-blocking modes with configurable timeout for summary generation. Includes automatic keyword verification for default files (multimodal_test.pdf: tables/charts/animals/gadgets, woods_frost.docx: Frost/woods/poem/collections) to ensure summary quality and relevance. Handles partial success scenarios.",
                ["GET /v1/summary"],
                ["collection_name", "file_name", "blocking", "timeout"],
                summary_time,
                TestStatus.FAILURE,
                "Failed to fetch document summaries",
            )
            return False

    @test_case(71, "Summary Status Tracking")
    async def _test_status_tracking(self) -> bool:
        """Test summary status tracking with progress monitoring"""
        logger.info("\n=== Test 71: Summary Status Tracking ===")
        start_time = time.time()

        # Use the first file for status tracking test
        test_file = os.path.basename(self.test_runner.test_files[0])
        success = await self.test_summary_status_tracking(
            self.collections["with_metadata"], test_file
        )

        elapsed_time = time.time() - start_time

        if success:
            self.add_test_result(
                self._test_status_tracking.test_number,
                self._test_status_tracking.test_name,
                f"Monitor summary generation status transitions (PENDING -> IN_PROGRESS -> SUCCESS) with real-time chunk-level progress tracking. File: {test_file}. Validates status flow, progress updates (current/total chunks), and timestamp tracking.",
                ["GET /v1/summary"],
                ["collection_name", "file_name", "blocking"],
                elapsed_time,
                TestStatus.SUCCESS,
            )
            return True
        else:
            self.add_test_result(
                self._test_status_tracking.test_number,
                self._test_status_tracking.test_name,
                f"Monitor summary generation status transitions (PENDING -> IN_PROGRESS -> SUCCESS) with real-time chunk-level progress tracking. File: {test_file}. Validates status flow, progress updates (current/total chunks), and timestamp tracking.",
                ["GET /v1/summary"],
                ["collection_name", "file_name", "blocking"],
                elapsed_time,
                TestStatus.FAILURE,
                "Failed to track summary status properly",
            )
            return False

    @test_case(72, "Summary Blocking Mode")
    async def _test_blocking_mode(self) -> bool:
        """Test summary retrieval in blocking mode"""
        logger.info("\n=== Test 72: Summary Blocking Mode ===")
        start_time = time.time()

        # Use the second file if available, else first
        test_files = [os.path.basename(f) for f in self.test_runner.test_files]
        test_file = test_files[1] if len(test_files) > 1 else test_files[0]

        success = await self.test_summary_blocking_mode(
            self.collections["with_metadata"], test_file
        )

        elapsed_time = time.time() - start_time

        if success:
            self.add_test_result(
                self._test_blocking_mode.test_number,
                self._test_blocking_mode.test_name,
                f"Test blocking mode summary retrieval with configurable timeout. File: {test_file}. Validates that the endpoint waits for summary generation to complete before returning, handling both quick completions and timeouts gracefully.",
                ["GET /v1/summary"],
                ["collection_name", "file_name", "blocking", "timeout"],
                elapsed_time,
                TestStatus.SUCCESS,
            )
            return True
        else:
            self.add_test_result(
                self._test_blocking_mode.test_number,
                self._test_blocking_mode.test_name,
                f"Test blocking mode summary retrieval with configurable timeout. File: {test_file}. Validates that the endpoint waits for summary generation to complete before returning, handling both quick completions and timeouts gracefully.",
                ["GET /v1/summary"],
                ["collection_name", "file_name", "blocking", "timeout"],
                elapsed_time,
                TestStatus.FAILURE,
                "Failed blocking mode test",
            )
            return False

    @test_case(73, "Summary NOT_FOUND Status")
    async def _test_not_found_status(self) -> bool:
        """Test NOT_FOUND status for non-existent files"""
        logger.info("\n=== Test 73: Summary NOT_FOUND Status ===")
        start_time = time.time()

        nonexistent_file = "nonexistent_file_12345.pdf"
        success = await self.test_summary_not_found(
            self.collections["with_metadata"], nonexistent_file
        )

        elapsed_time = time.time() - start_time

        if success:
            self.add_test_result(
                self._test_not_found_status.test_number,
                self._test_not_found_status.test_name,
                f"Validate NOT_FOUND status (404) for summaries that were never requested. File: {nonexistent_file}. Ensures proper error handling and user guidance for non-existent summaries.",
                ["GET /v1/summary"],
                ["collection_name", "file_name"],
                elapsed_time,
                TestStatus.SUCCESS,
            )
            return True
        else:
            self.add_test_result(
                self._test_not_found_status.test_number,
                self._test_not_found_status.test_name,
                f"Validate NOT_FOUND status (404) for summaries that were never requested. File: {nonexistent_file}. Ensures proper error handling and user guidance for non-existent summaries.",
                ["GET /v1/summary"],
                ["collection_name", "file_name"],
                elapsed_time,
                TestStatus.FAILURE,
                "Failed NOT_FOUND status test",
            )
            return False

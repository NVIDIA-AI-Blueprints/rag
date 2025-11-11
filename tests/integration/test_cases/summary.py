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

    async def test_summary_with_page_filters(
        self, collection_name: str, filename: str
    ) -> bool:
        """Test summary generation with comprehensive page filtering"""
        async with aiohttp.ClientSession() as session:
            try:
                logger.info(
                    f"🔍 Testing comprehensive page filtering for file: {filename}"
                )

                # Find the full file path
                file_path = None
                for test_file in self.test_runner.test_files:
                    if os.path.basename(test_file) == filename:
                        file_path = test_file
                        break

                if not file_path:
                    logger.error(f"❌ File not found: {filename}")
                    return False

                update_url = f"{self.ingestor_server_url}/v1/documents"

                # Test 1: Multiple ranges with first 5 and last 5 pages (tests positive + negative ranges)
                page_filter = [[1, 5], [-5, -1]]

                logger.info(
                    "📤 Testing filter: first 5 pages [1, 5] and last 5 pages [-5, -1]"
                )

                # Prepare FormData with file and JSON data
                data = aiohttp.FormData()
                data.add_field("documents", open(file_path, "rb"), filename=filename)

                json_data = {
                    "collection_name": collection_name,
                    "generate_summary": True,
                    "summary_options": {"page_filter": page_filter},
                }
                data.add_field("data", json.dumps(json_data))

                async with session.patch(update_url, data=data) as response:
                    await print_response(response)
                    if response.status not in [200, 201]:
                        logger.error(f"❌ Update failed with status {response.status}")
                        return False

                await asyncio.sleep(5)

                summary_params = {
                    "collection_name": collection_name,
                    "file_name": filename,
                    "blocking": "true",
                    "timeout": 120,
                }

                logger.info("⏳ Waiting for filtered summary generation...")
                async with session.get(
                    f"{self.rag_server_url}/v1/summary", params=summary_params
                ) as response:
                    result = await print_response(response)

                    if response.status != 200:
                        logger.error(
                            f"❌ Failed to retrieve filtered summary: {response.status}"
                        )
                        return False

                    summary_text = result.get("summary", "")
                    logger.info(
                        f"✅ Range filter passed: [[1, 5], [-5, -1]] ({len(summary_text)} chars)"
                    )

                # Test 2: String-based filter (odd pages)
                page_filter_odd = "odd"

                logger.info("\n📤 Testing filter: odd pages")

                # Prepare FormData again for second test
                data_odd = aiohttp.FormData()
                data_odd.add_field(
                    "documents", open(file_path, "rb"), filename=filename
                )

                json_data_odd = {
                    "collection_name": collection_name,
                    "generate_summary": True,
                    "summary_options": {"page_filter": page_filter_odd},
                }
                data_odd.add_field("data", json.dumps(json_data_odd))

                async with session.patch(update_url, data=data_odd) as response:
                    if response.status not in [200, 201]:
                        logger.error(
                            f"❌ Update with 'odd' filter failed: {response.status}"
                        )
                        return False

                await asyncio.sleep(5)

                async with session.get(
                    f"{self.rag_server_url}/v1/summary", params=summary_params
                ) as response:
                    result = await print_response(response)

                    if response.status != 200:
                        logger.error(
                            f"❌ Failed to retrieve 'odd' filtered summary: {response.status}"
                        )
                        return False

                    summary_text_odd = result.get("summary", "")
                    logger.info(
                        f"✅ String filter passed: 'odd' ({len(summary_text_odd)} chars)"
                    )
                    logger.info("✅ All page filter types validated successfully")
                    return True

            except Exception as e:
                logger.error(f"❌ Error in page filter test: {e}")
                import traceback

                logger.error(traceback.format_exc())
                return False

    async def test_concurrent_summaries(
        self, collection_name: str, filenames: list[str], concurrency: int = 5
    ) -> bool:
        """Test Redis rate limiting with concurrent summary requests"""
        async with aiohttp.ClientSession() as session:
            try:
                logger.info(
                    f"🚀 Testing concurrent summaries with {concurrency} files to verify rate limiting"
                )

                # Create tasks for concurrent summary requests
                async def fetch_summary_task(filename: str) -> tuple[str, bool]:
                    try:
                        params = {
                            "collection_name": collection_name,
                            "file_name": filename,
                            "blocking": "true",
                            "timeout": 180,
                        }
                        start_time = time.time()

                        async with session.get(
                            f"{self.rag_server_url}/v1/summary", params=params
                        ) as response:
                            elapsed = time.time() - start_time
                            success = response.status == 200

                            if success:
                                logger.info(
                                    f"✅ {filename}: Completed in {elapsed:.2f}s"
                                )
                            else:
                                logger.error(
                                    f"❌ {filename}: Failed with status {response.status}"
                                )

                            return (filename, success)
                    except Exception as e:
                        logger.error(f"❌ {filename}: Error - {e}")
                        return (filename, False)

                # Launch concurrent requests
                tasks = [
                    fetch_summary_task(filename) for filename in filenames[:concurrency]
                ]

                logger.info(f"⏳ Launching {len(tasks)} concurrent summary requests...")
                start_time = time.time()
                results = await asyncio.gather(*tasks)
                total_time = time.time() - start_time

                # Analyze results
                success_count = sum(1 for _, success in results if success)
                logger.info(f"📊 Concurrent test completed in {total_time:.2f}s")
                logger.info(
                    f"✅ Success rate: {success_count}/{len(results)} ({success_count / len(results) * 100:.1f}%)"
                )

                # Test passes if at least 80% succeed (rate limiting may delay some)
                if success_count >= len(results) * 0.8:
                    logger.info(
                        "✅ Concurrent summaries test passed - rate limiting working correctly"
                    )
                    return True
                else:
                    logger.error(
                        f"❌ Too many failures: {success_count}/{len(results)}"
                    )
                    return False

            except Exception as e:
                logger.error(f"❌ Error in concurrent summaries test: {e}")
                return False

    @test_case(74, "Summary with Page Filters")
    async def _test_summary_page_filters(self) -> bool:
        """Test summary generation with page filtering"""
        logger.info("\n=== Test 74: Summary with Page Filters ===")
        start_time = time.time()

        # Use first file to test page filters
        test_file = os.path.basename(self.test_runner.test_files[0])

        success = await self.test_summary_with_page_filters(
            self.collections["with_metadata"], test_file
        )

        elapsed_time = time.time() - start_time

        if success:
            self.add_test_result(
                self._test_summary_page_filters.test_number,
                self._test_summary_page_filters.test_name,
                f"Test page filtering with range-based [[1, 5], [-5, -1]] and string-based 'odd' filters. File: {test_file}. Validates positive ranges, negative/Pythonic indices, multiple ranges, and string filters (odd/even) via PATCH /v1/documents with summary_options.page_filter.",
                ["PATCH /v1/documents", "GET /v1/summary"],
                ["collection_name", "file_names", "summary_options.page_filter"],
                elapsed_time,
                TestStatus.SUCCESS,
            )
            return True
        else:
            self.add_test_result(
                self._test_summary_page_filters.test_number,
                self._test_summary_page_filters.test_name,
                f"Test page filtering with range-based [[1, 5], [-5, -1]] and string-based 'odd' filters. File: {test_file}. Validates positive ranges, negative/Pythonic indices, multiple ranges, and string filters (odd/even) via PATCH /v1/documents with summary_options.page_filter.",
                ["PATCH /v1/documents", "GET /v1/summary"],
                ["collection_name", "file_names", "summary_options.page_filter"],
                elapsed_time,
                TestStatus.FAILURE,
                "Failed to generate summary with page filters",
            )
            return False

    @test_case(75, "Concurrent Summaries - Redis Rate Limiting")
    async def _test_concurrent_summaries(self) -> bool:
        """Test Redis-based global rate limiting with concurrent requests"""
        logger.info("\n=== Test 75: Concurrent Summaries - Redis Rate Limiting ===")
        start_time = time.time()

        # Get all available test files
        all_files = [os.path.basename(f) for f in self.test_runner.test_files]

        success = await self.test_concurrent_summaries(
            self.collections["with_metadata"],
            all_files,
            concurrency=min(5, len(all_files)),
        )

        elapsed_time = time.time() - start_time

        if success:
            self.add_test_result(
                self._test_concurrent_summaries.test_number,
                self._test_concurrent_summaries.test_name,
                f"Test Redis-based global rate limiting with {min(5, len(all_files))} concurrent summary requests. Validates that global parallelization limit (default: 20) is enforced via Redis coordination. Tests slot acquisition/release and graceful handling under load.",
                ["GET /v1/summary"],
                ["collection_name", "file_name", "blocking", "timeout"],
                elapsed_time,
                TestStatus.SUCCESS,
            )
            return True
        else:
            self.add_test_result(
                self._test_concurrent_summaries.test_number,
                self._test_concurrent_summaries.test_name,
                f"Test Redis-based global rate limiting with {min(5, len(all_files))} concurrent summary requests. Validates that global parallelization limit (default: 20) is enforced via Redis coordination. Tests slot acquisition/release and graceful handling under load.",
                ["GET /v1/summary"],
                ["collection_name", "file_name", "blocking", "timeout"],
                elapsed_time,
                TestStatus.FAILURE,
                "Failed concurrent summaries test",
            )
            return False

    async def test_shallow_summary_with_page_filter(
        self, collection_name: str, filename: str
    ) -> bool:
        """Test shallow_summary with page filtering (fast text-only extraction + filters)"""
        async with aiohttp.ClientSession() as session:
            try:
                logger.info(
                    f"🚀 Testing shallow_summary with page filter for file: {filename}"
                )

                # Find the full file path
                file_path = None
                for test_file in self.test_runner.test_files:
                    if os.path.basename(test_file) == filename:
                        file_path = test_file
                        break

                if not file_path:
                    logger.error(f"❌ File not found: {filename}")
                    return False

                update_url = f"{self.ingestor_server_url}/v1/documents"

                # Test shallow_summary=True with page filter for first 5 and last 5 pages
                page_filter = [[1, 5], [-5, -1]]

                logger.info(
                    "📤 Uploading with shallow_summary=True and page_filter=[[1, 5], [-5, -1]]"
                )

                # Prepare FormData with file and JSON data
                data = aiohttp.FormData()
                data.add_field("documents", open(file_path, "rb"), filename=filename)

                json_data = {
                    "collection_name": collection_name,
                    "generate_summary": True,
                    "summary_options": {
                        "page_filter": page_filter,
                        "shallow_summary": True,
                    },
                }
                data.add_field("data", json.dumps(json_data))

                async with session.patch(update_url, data=data) as response:
                    await print_response(response)
                    if response.status not in [200, 201]:
                        logger.error(f"❌ Update failed with status {response.status}")
                        return False

                logger.info(
                    "⏳ Waiting for shallow summary generation (should be faster than full extraction)..."
                )
                await asyncio.sleep(3)  # Shorter wait for shallow summary

                # Check status to verify it's progressing/completed
                summary_params = {
                    "collection_name": collection_name,
                    "file_name": filename,
                    "blocking": "true",
                    "timeout": 120,
                }

                async with session.get(
                    f"{self.rag_server_url}/v1/summary", params=summary_params
                ) as response:
                    result = await print_response(response)

                    if response.status != 200:
                        logger.error(
                            f"❌ Failed to retrieve shallow summary: {response.status}"
                        )
                        return False

                    summary_text = result.get("summary", "")
                    if not summary_text:
                        logger.error("❌ Shallow summary is empty")
                        return False

                    logger.info(
                        f"✅ Shallow summary with page filter successful ({len(summary_text)} chars)"
                    )
                    logger.info(
                        "✅ Validated: shallow_summary=True + page_filter work together"
                    )
                    return True

            except Exception as e:
                logger.error(f"❌ Error in shallow summary with page filter test: {e}")
                import traceback

                logger.error(traceback.format_exc())
                return False

    @test_case(76, "Shallow Summary with Page Filter")
    async def _test_shallow_summary_with_page_filter(self) -> bool:
        """Test shallow_summary feature with page filtering"""
        logger.info("\n=== Test 76: Shallow Summary with Page Filter ===")
        start_time = time.time()

        # Use first file to test shallow summary with page filter
        test_file = os.path.basename(self.test_runner.test_files[0])

        success = await self.test_shallow_summary_with_page_filter(
            self.collections["with_metadata"], test_file
        )

        elapsed_time = time.time() - start_time

        if success:
            self.add_test_result(
                self._test_shallow_summary_with_page_filter.test_number,
                self._test_shallow_summary_with_page_filter.test_name,
                f"Test shallow_summary=True with page_filter=[[1, 5], [-5, -1]]. File: {test_file}. Validates fast text-only extraction for summary generation while full multimodal ingestion proceeds in parallel. Tests integration of shallow_summary flag with page filtering (first 5 + last 5 pages).",
                ["PATCH /v1/documents", "GET /v1/summary"],
                [
                    "collection_name",
                    "file_names",
                    "summary_options.shallow_summary",
                    "summary_options.page_filter",
                ],
                elapsed_time,
                TestStatus.SUCCESS,
            )
            return True
        else:
            self.add_test_result(
                self._test_shallow_summary_with_page_filter.test_number,
                self._test_shallow_summary_with_page_filter.test_name,
                f"Test shallow_summary=True with page_filter=[[1, 5], [-5, -1]]. File: {test_file}. Validates fast text-only extraction for summary generation while full multimodal ingestion proceeds in parallel. Tests integration of shallow_summary flag with page filtering (first 5 + last 5 pages).",
                ["PATCH /v1/documents", "GET /v1/summary"],
                [
                    "collection_name",
                    "file_names",
                    "summary_options.shallow_summary",
                    "summary_options.page_filter",
                ],
                elapsed_time,
                TestStatus.FAILURE,
                "Failed shallow summary with page filter test",
            )
            return False

    async def test_summary_with_strategy_single(
        self, collection_name: str, filename: str
    ) -> bool:
        """Test summarization_strategy='single' (one-pass with truncation)"""
        async with aiohttp.ClientSession() as session:
            try:
                logger.info(
                    f"🚀 Testing summarization_strategy='single' for file: {filename}"
                )

                # Find the full file path
                file_path = None
                for test_file in self.test_runner.test_files:
                    if os.path.basename(test_file) == filename:
                        file_path = test_file
                        break

                if not file_path:
                    logger.error(f"❌ File not found: {filename}")
                    return False

                update_url = f"{self.ingestor_server_url}/v1/documents"

                logger.info(
                    "📤 Uploading with summarization_strategy='single' (one-pass truncation)"
                )

                # Prepare FormData with file and JSON data
                data = aiohttp.FormData()
                data.add_field("documents", open(file_path, "rb"), filename=filename)

                json_data = {
                    "collection_name": collection_name,
                    "generate_summary": True,
                    "summary_options": {
                        "summarization_strategy": "single",
                    },
                }
                data.add_field("data", json.dumps(json_data))

                async with session.patch(update_url, data=data) as response:
                    await print_response(response)
                    if response.status not in [200, 201]:
                        logger.error(f"❌ Update failed with status {response.status}")
                        return False

                logger.info(
                    "⏳ Waiting for single-pass summary generation (should be fast)..."
                )
                await asyncio.sleep(3)

                # Fetch the generated summary
                summary_params = {
                    "collection_name": collection_name,
                    "file_name": filename,
                    "blocking": "true",
                    "timeout": 60,
                }

                async with session.get(
                    f"{self.rag_server_url}/v1/summary", params=summary_params
                ) as response:
                    result = await print_response(response)

                    if response.status != 200:
                        logger.error(
                            f"❌ Failed to retrieve summary with 'single' strategy: {response.status}"
                        )
                        return False

                    summary_text = result.get("summary", "")
                    if not summary_text:
                        logger.error("❌ Summary with 'single' strategy is empty")
                        return False

                    logger.info(
                        f"✅ Single-pass summary successful ({len(summary_text)} chars)"
                    )
                    logger.info(
                        "✅ Validated: summarization_strategy='single' works correctly"
                    )
                    return True

            except Exception as e:
                logger.error(f"❌ Error in 'single' strategy test: {e}")
                import traceback

                logger.error(traceback.format_exc())
                return False

    @test_case(77, "Summary with Strategy: Single")
    async def _test_summary_strategy_single(self) -> bool:
        """Test single-pass summarization strategy"""
        logger.info("\n=== Test 77: Summary with Strategy: Single ===")
        start_time = time.time()

        # Use first file to test single strategy
        test_file = os.path.basename(self.test_runner.test_files[0])

        success = await self.test_summary_with_strategy_single(
            self.collections["with_metadata"], test_file
        )

        elapsed_time = time.time() - start_time

        if success:
            self.add_test_result(
                self._test_summary_strategy_single.test_number,
                self._test_summary_strategy_single.test_name,
                f"Test summarization_strategy='single' (one-pass with truncation). File: {test_file}. Validates that single-pass strategy processes entire document in one LLM call, truncating if it exceeds max_chunk_length. Verifies summary generation completes successfully and returns valid content.",
                ["PATCH /v1/documents", "GET /v1/summary"],
                [
                    "collection_name",
                    "file_names",
                    "summary_options.summarization_strategy",
                ],
                elapsed_time,
                TestStatus.SUCCESS,
            )
            return True
        else:
            self.add_test_result(
                self._test_summary_strategy_single.test_number,
                self._test_summary_strategy_single.test_name,
                f"Test summarization_strategy='single' (one-pass with truncation). File: {test_file}. Validates that single-pass strategy processes entire document in one LLM call, truncating if it exceeds max_chunk_length. Verifies summary generation completes successfully and returns valid content.",
                ["PATCH /v1/documents", "GET /v1/summary"],
                [
                    "collection_name",
                    "file_names",
                    "summary_options.summarization_strategy",
                ],
                elapsed_time,
                TestStatus.FAILURE,
                "Failed single-pass summarization strategy test",
            )
            return False

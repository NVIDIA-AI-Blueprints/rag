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

#!/usr/bin/env python3
"""
Modular Integration Test Script for RAG and Ingestion APIs

This script performs end-to-end testing of the RAG and ingestion microservices,
testing all major API endpoints and workflows using a modular architecture.

Usage:
    python main.py --help
    python main.py --rag-server http://localhost:8081 --ingestor-server http://localhost:8082
    python main.py --rag-server http://localhost:8081 --ingestor-server http://localhost:8082 --collection-with-metadata my_collection --files-with-metadata file1.pdf file2.docx
"""

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Any

import aiohttp

from .base import TestResult, TestStatus
from .utils.discovery import discover_test_modules, discover_test_cases
from .utils.sequence_executor import TestSequenceExecutor
from .utils.response_handlers import print_response

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("integration_test.log"),
    ],
)
logger = logging.getLogger(__name__)


class IntegrationTestRunner:
    """Integration test runner for RAG and Ingestion APIs"""

    def __init__(
        self,
        rag_server_url: str,
        ingestor_server_url: str,
        data_dir: str,
        timeout: int = 300,
        poll_interval: int = 5,
        collection_with_metadata: str | None = None,
        collection_without_metadata: str | None = None,
        files_with_metadata: list[str] | None = None,
        files_without_metadata: list[str] | None = None,
    ):
        self.rag_server_url = rag_server_url.rstrip("/")
        self.ingestor_server_url = ingestor_server_url.rstrip("/")
        self.data_dir = Path(data_dir)
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.test_files = []

        # Use provided collection names or defaults
        self.collections = {
            "with_metadata": collection_with_metadata
            or "test_collection_with_metadata",
            "without_metadata": collection_without_metadata
            or "test_collection_without_metadata",
        }

        # Store file specifications
        self.files_with_metadata = files_with_metadata or []
        self.files_without_metadata = files_without_metadata or []



        # Auto-discovery of test modules and cases
        self.test_modules = discover_test_modules()
        self.test_registry = discover_test_cases()
        self.sequence_executor = TestSequenceExecutor()

        # Build reverse mapping for test names
        self.test_name_to_number = {}
        for test_num, (module_class, method_name, test_name) in self.test_registry.items():
            self.test_name_to_number[test_name] = test_num

        # Track current test phase for result tracking
        self.current_test_phase = "main"

        # Expected metadata schema for validation
        self.expected_metadata_schema = [
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

        # Metadata configuration for testing
        self.original_metadata = {
            "timestamp": "2024-01-15T10:23:00Z",
            "meta_field_1": "multimodal document",
        }

        self.updated_metadata = {
            "timestamp": "2024-01-16T10:23:00Z",
            "meta_field_1": "updated multimodal document",
        }

        self.metadata_filter_expr = (
            'content_metadata["meta_field_1"] == "updated multimodal document"'
        )

        self.task_ids = {}
        self.test_results: list[TestResult] = []



    def add_test_result(
        self,
        test_number: int,
        test_name: str,
        description: str,
        api_endpoints: list[str],
        payload_fields: list[str],
        time_taken: float,
        status: TestStatus,
        error_message: str | None = None,
        test_phase: str | None = None,
    ):
        """Add a test result to the results list"""
        # Use current_test_phase if no specific phase is provided
        if test_phase is None:
            test_phase = self.current_test_phase

        result = TestResult(
            test_number=test_number,
            test_name=test_name,
            description=description,
            api_endpoints=api_endpoints,
            payload_fields=payload_fields,
            time_taken=time_taken,
            status=status,
            error_message=error_message,
            test_phase=test_phase,
        )
        self.test_results.append(result)

    async def print_response(self, response: aiohttp.ClientResponse) -> dict[str, Any]:
        """Helper to print API response and return JSON"""
        return await print_response(response)

    async def run_integration_tests(self, sequence_name=None, test_numbers=None, test_names=None, test_range=None, exclude_tests=None) -> bool:
        """Run integration tests with optional test selection or sequence execution"""
        logger.info("🚀 Starting RAG and Ingestion API Integration Tests")

        # Determine which tests to run
        if sequence_name:
            # Run a predefined sequence
            if not self.sequence_executor.validate_sequence(sequence_name):
                return False

            test_numbers = self.sequence_executor.get_sequence_test_numbers(sequence_name)
            sequence_info = self.sequence_executor.get_sequence_info(sequence_name)
            logger.info(f"🎯 Running sequence '{sequence_name}': {sequence_info['name']}")
            logger.info(f"📋 Description: {sequence_info['description']}")
            logger.info(f"🔢 Test numbers: {test_numbers}")
        elif not test_numbers and not test_names and not test_range:
            # Default to basic sequence if no test selection arguments provided
            sequence_name = "basic"
            if not self.sequence_executor.validate_sequence(sequence_name):
                return False

            test_numbers = self.sequence_executor.get_sequence_test_numbers(sequence_name)
            sequence_info = self.sequence_executor.get_sequence_info(sequence_name)
            logger.info(f"🎯 Running default sequence '{sequence_name}': {sequence_info['name']}")
            logger.info(f"📋 Description: {sequence_info['description']}")
            logger.info(f"🔢 Test numbers: {test_numbers}")

        # Parse test identifiers
        include_tests = set()
        ordered_test_numbers = []  # Preserve order for sequence tests

        if test_numbers:
            include_tests.update(test_numbers)
            ordered_test_numbers = test_numbers  # Preserve the original order

        if test_names:
            name_tests = self._parse_test_identifiers(test_names)
            include_tests.update(name_tests)

        # Parse test range if provided
        if test_range:
            try:
                start, end = map(int, test_range.split('-'))
                range_tests = set(range(start, end + 1))
                include_tests.update(range_tests)
            except ValueError:
                logger.error(f"Invalid test range format: {test_range}. Expected format: start-end (e.g., 1-5)")
                return False

        # Convert to set for efficient lookup
        include_tests = include_tests if include_tests else None
        exclude_tests = set(exclude_tests) if exclude_tests else set()

        # Log which tests will be run
        if include_tests:
            test_names_to_run = []
            for num in sorted(include_tests):
                if num in self.test_registry:
                    _, _, test_name = self.test_registry[num]
                    test_names_to_run.append(test_name)
                else:
                    test_names_to_run.append(f"Test {num}")
            logger.info(f"🎯 Running selected tests: {', '.join(test_names_to_run)}")
        if exclude_tests:
            test_names_to_exclude = []
            for num in sorted(exclude_tests):
                if num in self.test_registry:
                    _, _, test_name = self.test_registry[num]
                    test_names_to_exclude.append(test_name)
                else:
                    test_names_to_exclude.append(f"Test {num}")
            logger.info(f"🚫 Excluding tests: {', '.join(test_names_to_exclude)}")

        try:
            # Execute pre-sequence tests if running a sequence
            if sequence_name:
                pre_tests = self.sequence_executor.get_pre_sequence_tests(sequence_name)
                if pre_tests:
                    logger.info(f"🔧 Running pre-sequence tests: {pre_tests}")
                    pre_success = await self._execute_specific_tests(pre_tests, set(), "pre-sequence")
                    if not pre_success:
                        logger.error("❌ Pre-sequence tests failed")
                        return False

            # Execute tests based on the test registry
            if include_tests:
                # Run specific tests - use ordered test numbers if available (for sequences)
                if ordered_test_numbers:
                    # Filter ordered_test_numbers to only include tests that are in include_tests
                    tests_to_run = [num for num in ordered_test_numbers if num in include_tests]
                    success = await self._execute_specific_tests(tests_to_run, exclude_tests, "main")
                else:
                    # Fall back to sorted order for non-sequence tests
                    success = await self._execute_specific_tests(sorted(include_tests), exclude_tests, "main")
            else:
                # Run all tests (default behavior)
                all_test_numbers = sorted(self.test_registry.keys())
                success = await self._execute_specific_tests(all_test_numbers, exclude_tests, "main")

            # Execute post-sequence tests if running a sequence (always run, even if main sequence fails)
            if sequence_name:
                post_tests = self.sequence_executor.get_post_sequence_tests(sequence_name)
                if post_tests:
                    logger.info(f"🧹 Running post-sequence tests: {post_tests}")
                    # Execute post-sequence tests but don't fail the entire sequence if they fail
                    post_success = await self._execute_specific_tests(post_tests, set(), "post-sequence")
                    if not post_success:
                        logger.warning("⚠️ Some post-sequence tests failed, but continuing...")

            if success:
                logger.info("🎉 All integration tests completed!")
            return success
        except Exception as e:
            logger.error(f"❌ Unexpected error during tests: {e}")
            # Execute post-sequence tests even on error if running a sequence
            if sequence_name:
                post_tests = self.sequence_executor.get_post_sequence_tests(sequence_name)
                if post_tests:
                    logger.info(f"🧹 Running post-sequence tests after error: {post_tests}")
                    # Execute post-sequence tests but don't fail the entire sequence if they fail
                    post_success = await self._execute_specific_tests(post_tests, set(), "post-sequence")
                    if not post_success:
                        logger.warning("⚠️ Some post-sequence tests failed after error, but continuing...")
            return False

    async def _execute_specific_tests(self, test_numbers: list[int], exclude_tests: set[int], test_phase: str = "main") -> bool:
        """Execute specific tests based on test numbers"""
        # Set current test phase for result tracking
        self.current_test_phase = test_phase

        # Create module instances for all test modules
        module_instances = {}
        for module_class in self.test_modules:
            module_instances[module_class] = module_class(self)

        # Execute tests in order
        for test_num in test_numbers:
            if test_num in exclude_tests:
                logger.info(f"⏭️ Skipping test {test_num} (excluded)")
                continue

            if test_num not in self.test_registry:
                logger.warning(f"⚠️ Test {test_num} not found in registry, skipping")
                continue

            module_class, method_name, test_name = self.test_registry[test_num]
            module_instance = module_instances[module_class]

            # Add phase indicator to log message
            phase_indicator = ""
            if test_phase == "pre-sequence":
                phase_indicator = "🔧 [PRE-SEQUENCE] "
            elif test_phase == "post-sequence":
                phase_indicator = "🧹 [POST-SEQUENCE] "

            logger.info(f"\n=== {phase_indicator}Test {test_num}: {test_name} ===")

            try:
                # Get the test method
                test_method = getattr(module_instance, method_name)

                # Execute the test
                success = await test_method()

                if not success:
                    logger.error(f"❌ Test {test_num} ({test_name}) failed")
                    # For post-sequence tests, continue executing other tests even if one fails
                    if test_phase == "post-sequence":
                        logger.warning(f"⚠️ Test {test_num} failed in post-sequence, but continuing with remaining tests...")
                        continue
                    else:
                        return False

            except Exception as e:
                logger.error(f"❌ Test {test_num} ({test_name}) failed with exception: {e}")
                # For post-sequence tests, continue executing other tests even if one fails
                if test_phase == "post-sequence":
                    logger.warning(f"⚠️ Test {test_num} failed with exception in post-sequence, but continuing with remaining tests...")
                    continue
                else:
                    return False

        return True



    def print_test_results_table(self):
        """Print a comprehensive test results table"""
        logger.info("\n" + "=" * 120)
        logger.info("INTEGRATION TEST RESULTS SUMMARY")
        logger.info("=" * 120)

        # Check if there are any test results
        if not self.test_results:
            logger.info("No test results to display.")
            return

        # Prepare table data
        table_data = []
        for result in self.test_results:
            # Format API endpoints and payload fields for better readability
            api_endpoints_str = "\n".join(
                [f"• {endpoint}" for endpoint in result.api_endpoints]
            )
            payload_fields_str = "\n".join(
                [f"• {field}" for field in result.payload_fields]
            )

            # Format status with color indicators
            status_str = result.status.value
            if result.status == TestStatus.SUCCESS:
                status_str = "✅ " + status_str
            elif result.status == TestStatus.FAILURE:
                status_str = "❌ " + status_str
            else:
                status_str = "⏸️ " + status_str

            # Add error message if present
            description_with_error = result.description
            if result.error_message:
                description_with_error += "\n\nError: " + result.error_message

            # Format test phase with indicators
            phase_str = ""
            if result.test_phase == "pre-sequence":
                phase_str = "🔧 PRE"
            elif result.test_phase == "post-sequence":
                phase_str = "🧹 POST"
            else:
                phase_str = "📋 MAIN"

            table_data.append(
                [
                    f"Test {result.test_number}",
                    result.test_name,
                    description_with_error,
                    api_endpoints_str,
                    payload_fields_str,
                    f"{result.time_taken:.2f}s",
                    status_str,
                    phase_str,
                ]
            )

        # Create table headers
        headers = [
            "Test #",
            "Test Case",
            "Description",
            "API Endpoints",
            "Payload Fields",
            "Time Taken",
            "Status",
            "Phase",
        ]

        # Print the table
        from tabulate import tabulate
        table = tabulate(
            table_data,
            headers=headers,
            tablefmt="grid",
            maxcolwidths=[8, 20, 25, 20, 20, 10, 15, 8],
            stralign="left",
        )

        print(table)

        # Print summary statistics
        total_tests = len(self.test_results)
        successful_tests = sum(
            1 for r in self.test_results if r.status == TestStatus.SUCCESS
        )
        failed_tests = sum(
            1 for r in self.test_results if r.status == TestStatus.FAILURE
        )
        not_executed_tests = sum(
            1 for r in self.test_results if r.status == TestStatus.NOT_EXECUTED
        )
        total_time = sum(r.time_taken for r in self.test_results)

        # Phase breakdown
        pre_sequence_tests = sum(1 for r in self.test_results if r.test_phase == "pre-sequence")
        main_tests = sum(1 for r in self.test_results if r.test_phase == "main")
        post_sequence_tests = sum(1 for r in self.test_results if r.test_phase == "post-sequence")

        logger.info("\nSUMMARY STATISTICS:")
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Successful: {successful_tests}")
        logger.info(f"Failed: {failed_tests}")
        logger.info(f"Not Executed: {not_executed_tests}")
        logger.info(f"Total Time: {total_time:.2f}s")
        logger.info(
            f"Success Rate: {(successful_tests / total_tests * 100):.1f}%"
            if total_tests > 0
            else "N/A"
        )

        # Phase breakdown
        if pre_sequence_tests > 0 or post_sequence_tests > 0:
            logger.info("\nPHASE BREAKDOWN:")
            if pre_sequence_tests > 0:
                logger.info(f"🔧 Pre-sequence Tests: {pre_sequence_tests}")
            if main_tests > 0:
                logger.info(f"📋 Main Tests: {main_tests}")
            if post_sequence_tests > 0:
                logger.info(f"🧹 Post-sequence Tests: {post_sequence_tests}")
        logger.info("=" * 120)

    def list_available_tests(self):
        """List all available tests with numbers and names"""
        logger.info("Available Tests:")
        logger.info("=" * 60)
        logger.info(f"{'Test #':<8} {'Test Name':<40}")
        logger.info("-" * 60)
        # Sort tests by test number for better readability
        for test_num in sorted(self.test_registry.keys()):
            module_class, method_name, test_name = self.test_registry[test_num]
            logger.info(f"{test_num:<8} {test_name:<40}")
        logger.info("=" * 60)
        logger.info("\nUsage Examples:")
        logger.info("  --test-numbers 1 3 5")
        logger.info("  --test-names 'Health Checks' 'Create Collections'")
        logger.info("  --tests 1 'Create Collections' 5")
        logger.info("  --test-names 'Health' 'Upload'  # Partial name matching)")
        logger.info("  --test-range 1-5")
        logger.info("  --exclude-tests 16")

    def _parse_test_identifiers(self, test_identifiers):
        """Parse test identifiers (numbers or names) into test numbers"""
        test_numbers = set()

        for identifier in test_identifiers:
            # Try to convert to integer (test number)
            try:
                test_num = int(identifier)
                if test_num in self.test_registry:
                    test_numbers.add(test_num)
                else:
                    logger.warning(f"Test number {test_num} not found in registry")
            except ValueError:
                # Not a number, try to match by name
                matched_tests = self._find_tests_by_name(identifier)
                if matched_tests:
                    test_numbers.update(matched_tests)
                else:
                    logger.warning(f"No tests found matching name: {identifier}")

        return test_numbers

    def _find_tests_by_name(self, name_pattern):
        """Find test numbers by name pattern (supports partial matching)"""
        matched_tests = set()
        name_pattern_lower = name_pattern.lower()

        for test_num, test_name in self.test_registry.items():
            if name_pattern_lower in test_name.lower():
                matched_tests.add(test_num)

        return matched_tests

    def _should_run_test(self, test_number, include_tests=None, exclude_tests=None):
        """Determine if a test should run based on filters"""
        if exclude_tests and test_number in exclude_tests:
            return False

        if include_tests is None:
            return True

        return test_number in include_tests


def main():
    """Main function to run integration tests"""
    parser = argparse.ArgumentParser(
        description="Modular Integration Test Script for RAG and Ingestion APIs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with recommended default files
  python main.py --rag-server http://localhost:8081 --ingestor-server http://localhost:8082

  # Use specific data directory
  python main.py --rag-server http://rag-server:8081 --ingestor-server http://ingestor-server:8082 --data-dir ../../data/multimodal

  # Custom timeout and poll interval
  python main.py --timeout 600 --poll-interval 10

  # Use custom collection names
  python main.py --collection-with-metadata my_metadata_collection --collection-without-metadata my_plain_collection

  # Override default files with custom files
  python main.py --files-with-metadata doc1.pdf doc2.docx --files-without-metadata doc3.pdf

  # Combine custom collections and files
  python main.py --collection-with-metadata test_metadata --files-with-metadata /path/to/file1.pdf /path/to/file2.docx

  # Skip cleanup to preserve test collections and documents
  python main.py --no-cleanup

  # Test selection examples:
  # Run specific test numbers
  python main.py --test-numbers 1 3 5

  # Run tests by name
  python main.py --test-names "Health Checks" "Create Collections"

  # Run tests by partial name matching
  python main.py --test-names "Health" "Upload" "Search"

  # Mix numbers and names
  python main.py --tests 1 "Create Collections" 5

  # Run test ranges
  python main.py --test-range 1-5

  # Exclude specific tests
  python main.py --exclude-tests 16

  # List all available tests
  python main.py --list-tests

  # Sequence examples:
  # Run predefined sequences
  python main.py --sequence basic
  python main.py --sequence optional
  python main.py --sequence full

  # List all available sequences
  python main.py --list-sequences
        """,
    )

    parser.add_argument(
        "--rag-server",
        default="http://localhost:8081",
        help="RAG server URL (default: http://localhost:8081)",
    )

    parser.add_argument(
        "--ingestor-server",
        default="http://localhost:8082",
        help="Ingestor server URL (default: http://localhost:8082)",
    )

    parser.add_argument(
        "--data-dir",
        default="./data/multimodal",
        help="Directory containing test files (default: ./data/multimodal)",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout for task completion in seconds (default: 300)",
    )

    parser.add_argument(
        "--poll-interval",
        type=int,
        default=5,
        help="Poll interval for task status in seconds (default: 5)",
    )

    parser.add_argument(
        "--collection-with-metadata",
        help="Name for the collection with metadata schema for basic sequence (default: test_collection_with_metadata)",
    )

    parser.add_argument(
        "--collection-without-metadata",
        help="Name for the collection without metadata schema for basic sequence (default: test_collection_without_metadata)",
    )

    parser.add_argument(
        "--files-with-metadata",
        nargs="+",
        help="Specific files to use for the collection with metadata for basic sequence (default: multimodal_test.pdf, woods_frost.docx)",
    )

    parser.add_argument(
        "--files-without-metadata",
        nargs="+",
        help="Specific files to use for the collection without metadata for basic sequence (default: table_test.pdf, embedded_table.pdf)",
    )



    parser.add_argument(
        "--test-numbers",
        type=int,
        nargs="+",
        help="Run specific test numbers (e.g., --test-numbers 1 3 5)"
    )

    parser.add_argument(
        "--test-names",
        nargs="+",
        help="Run tests by name (e.g., --test-names 'Health Checks' 'Create Collections')"
    )

    parser.add_argument(
        "--tests",
        nargs="+",
        help="Run tests by number or name (e.g., --tests 1 'Create Collections' 5)"
    )

    parser.add_argument(
        "--test-range",
        help="Run test range (e.g., --test-range 1-5)"
    )

    parser.add_argument(
        "--exclude-tests",
        type=int,
        nargs="+",
        help="Exclude specific test numbers (e.g., --exclude-tests 16)"
    )

    parser.add_argument(
        "--sequence",
        help="Run a predefined test sequence"
    )

    parser.add_argument(
        "--list-sequences",
        action="store_true",
        help="List all available test sequences and exit"
    )

    parser.add_argument(
        "--list-tests",
        action="store_true",
        help="List all available tests and exit"
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate sequence argument if provided
    if args.sequence:
        # Create a temporary test runner to validate sequence
        temp_runner = IntegrationTestRunner(
            rag_server_url="http://localhost:8081",
            ingestor_server_url="http://localhost:8082",
            data_dir="./data/multimodal",
        )
        available_sequences = list(temp_runner.sequence_executor.get_available_sequences().keys())

        if args.sequence not in available_sequences:
            logger.error(f"Invalid sequence '{args.sequence}'. Available sequences: {', '.join(available_sequences)}")
            sys.exit(1)

    # Create test runner
    runner = IntegrationTestRunner(
        rag_server_url=args.rag_server,
        ingestor_server_url=args.ingestor_server,
        data_dir=args.data_dir,
        timeout=args.timeout,
        poll_interval=args.poll_interval,
        collection_with_metadata=args.collection_with_metadata,
        collection_without_metadata=args.collection_without_metadata,
        files_with_metadata=args.files_with_metadata,
        files_without_metadata=args.files_without_metadata,

    )

    # Handle --list-sequences argument
    if args.list_sequences:
        runner.sequence_executor.list_sequences()
        sys.exit(0)

    # Handle --list-tests argument
    if args.list_tests:
        runner.list_available_tests()
        sys.exit(0)

    # Parse test selection arguments
    test_numbers = args.test_numbers
    test_names = args.test_names
    test_range = args.test_range
    exclude_tests = args.exclude_tests

    # Handle --tests argument (mixed numbers and names)
    if args.tests:
        # Parse mixed identifiers
        mixed_tests = runner._parse_test_identifiers(args.tests)
        if test_numbers:
            test_numbers.extend(list(mixed_tests))
        else:
            test_numbers = list(mixed_tests)

    # Run tests
    try:
        success = asyncio.run(runner.run_integration_tests(
            sequence_name=args.sequence,
            test_numbers=test_numbers,
            test_names=test_names,
            test_range=test_range,
            exclude_tests=exclude_tests
        ))

        # Print test results table regardless of success/failure
        runner.print_test_results_table()

        if success:
            logger.info("✅ All tests passed successfully!")
            sys.exit(0)
        else:
            logger.error("❌ Some tests failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("🛑 Tests interrupted by user")
        # Print test results table even when interrupted
        runner.print_test_results_table()
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        # Print test results table even on unexpected error
        runner.print_test_results_table()
        sys.exit(1)


if __name__ == "__main__":
    main()
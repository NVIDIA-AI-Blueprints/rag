# RAG Integration Tests

This directory contains modular integration tests for RAG and Ingestion APIs, organized by functionality for easier maintenance and extension.

## Prerequisites

### 1. Running Services
- **RAG Server**: Must be running on `http://localhost:8081` (or custom URL)
- **Ingestor Server**: Must be running on `http://localhost:8082` (or custom URL)
- **Milvus**: Vector database must be accessible
- **Nvingest**: Ensure nvingest and its dependencies are also running.
- **Model Endpoints**: Required for RAG operations, can be cloud or on-prem.

### 2. Test Data
- **Recommended**: The test uses specific default files for optimal testing:
  - Collection with metadata: `multimodal_test.pdf`, `woods_frost.docx`
  - Collection without metadata: `table_test.pdf`, `embedded_table.pdf`
- These files should be placed in the data directory (default: `../../data/multimodal`)
- Ensure files are accessible and readable
- Alternatively, specify individual files using the file arguments to override defaults

**Recommended Files for Testing:**
- **Collection with metadata**: `multimodal_test.pdf`, `woods_frost.docx`
- **Collection without metadata**: `table_test.pdf`, `embedded_table.pdf`

These files should be placed in your data directory (default: `../../data/multimodal`).

> **Why These Files?** These specific files are chosen because they provide comprehensive test coverage including multimodal content, tables, embedded objects, and various document formats that exercise all major features of the RAG pipeline. The sample queries and response checks are also tailored to these files.

## Quick Start

### 1. Install Requirements
```bash
pip install -r tests/integration/requirements.txt
```

### 2. Run Tests
```bash
# Default: runs basic sequence (core functionality tests)
python -m tests.integration.main --rag-server http://localhost:8081 --ingestor-server http://localhost:8082

# Run specific sequences
python -m tests.integration.main --sequence basic
python -m tests.integration.main --sequence nemo_guardrails
python -m tests.integration.main --sequence custom_prompt

# List available sequences
python -m tests.integration.main --list-sequences
```

### 3. Additional Options
```bash
# With custom data directory
python -m tests.integration.main --data-dir ../../data/multimodal

# Override with custom files
python -m tests.integration.main --files-with-metadata doc1.pdf doc2.docx --files-without-metadata doc3.pdf

# Use custom collection names
python -m tests.integration.main --collection-with-metadata my_metadata_collection --collection-without-metadata my_plain_collection

# Verbose logging
python -m tests.integration.main --verbose

# Test selection examples
python -m tests.integration.main --test-numbers 1 3 5
python -m tests.integration.main --test-names "Health Checks" "Create Collections"
python -m tests.integration.main --test-range 1-5
python -m tests.integration.main --exclude-tests 16
python -m tests.integration.main --list-tests
```

### Sequence

Sequences are defined in `test_sequences.yaml` file.

Each sequence can define:
- **`pre_sequence`**: Tests to run before the main sequence (e.g., setup, cleanup)
- **`test_cases`**: Main test cases to run in sequence.
- **`post_sequence`**: Tests to run after the main sequence (e.g., cleanup, verification)

**Execution Flow:**
1. Pre-sequence tests run first (if defined)
2. Main sequence tests run defined by `test_cases: []` in the same order as specified.
3. Post-sequence tests run last (always executes, even if main sequence fails)

**Result Display:**
- Tests are automatically marked as 🔧 PRE, 📋 MAIN, or 🧹 POST in the results table
- Summary includes phase breakdown statistics

### Usage
```bash
# Default: runs basic sequence
python -m tests.integration.main

# Run specific sequence
python -m tests.integration.main --sequence basic
python -m tests.integration.main --sequence image_captioning
python -m tests.integration.main --sequence full

# List available sequences (shows pre/post hooks)
python -m tests.integration.main --list-sequences
```

## Adding New Test Cases

### Quick Guide: Adding a New Test Case

1. **Choose the appropriate test module** based on functionality
2. **Add your test method** with the `@test_case` decorator:

```python
from ..base import test_case

@test_case(26, "New Feature Test") # Ensure the test number is unique
async def _test_new_feature(self) -> bool:
    """Test new feature functionality"""
    try:
        # Prepare request payload
        payload = {
            "param1": "value1",
            "param2": "value2"
        }

        # Log request details
        logger.info(f"🔧 Testing new feature")
        logger.info(f"📋 Request payload:\n{json.dumps(payload, indent=2)}")

        # Your test logic here - implement the actual API calls
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.rag_server_url}/v1/your-endpoint", json=payload) as response:
                result = await response.json()
                if response.status == 200:
                    logger.info(f"✅ New feature test passed:")
                    logger.info(f"Response JSON:\n{json.dumps(result, indent=2)}")

                    # Add test result automatically
                    self.add_test_result(
                        self._test_new_feature.test_number,
                        self._test_new_feature.test_name,
                        "Tests the new feature functionality",
                        ["POST /v1/your-endpoint"],
                        ["param1", "param2"],
                        time.time() - start_time,
                        TestStatus.SUCCESS
                    )
                    return True
                else:
                    logger.error(f"❌ New feature test failed: {response.status}")
                    logger.error(f"Response JSON:\n{json.dumps(result, indent=2)}")
                    return False
    except Exception as e:
        logger.error(f"❌ New feature test error: {e}")
        return False
```

**That's it!** The test will be automatically discovered and can be run with:
```bash
python -m tests.integration.main --test-numbers 26
```

### Adding a New Test Module

If you need to create a completely new test module:

1. **Create new test module file** in `test_cases/` directory:

```python
# test_cases/new_feature.py
import json
import logging
import time
from typing import Any

import aiohttp

from ..base import BaseTestModule, TestStatus, test_case

logger = logging.getLogger(__name__)

class NewFeatureModule(BaseTestModule):
    @test_case(26, "New Feature Test 1")
    async def _test_feature_1(self) -> bool:
        """Test feature 1 - implement actual API calls"""
        try:
            payload = {"field1": "value1"}
            logger.info(f"🔧 Testing feature 1")
            logger.info(f"📋 Request payload:\n{json.dumps(payload, indent=2)}")

            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.rag_server_url}/v1/feature1", json=payload) as response:
                    result = await response.json()
                    if response.status == 200:
                        logger.info(f"✅ Feature 1 test passed:")
                        logger.info(f"Response JSON:\n{json.dumps(result, indent=2)}")
                        return True
                    else:
                        logger.error(f"❌ Feature 1 test failed: {response.status}")
                        logger.error(f"Response JSON:\n{json.dumps(result, indent=2)}")
                        return False
        except Exception as e:
            logger.error(f"❌ Feature 1 test error: {e}")
            return False

    @test_case(27, "New Feature Test 2")
    async def _test_feature_2(self) -> bool:
        """Test feature 2 - implement actual API calls"""
        try:
            logger.info(f"🔧 Testing feature 2")

            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.rag_server_url}/v1/feature2") as response:
                    result = await response.json()
                    if response.status == 200:
                        logger.info(f"✅ Feature 2 test passed:")
                        logger.info(f"Response JSON:\n{json.dumps(result, indent=2)}")
                        return True
                    else:
                        logger.error(f"❌ Feature 2 test failed: {response.status}")
                        logger.error(f"Response JSON:\n{json.dumps(result, indent=2)}")
                        return False
        except Exception as e:
            logger.error(f"❌ Feature 2 test error: {e}")
            return False
```

**That's it!** The module will be automatically discovered. No need to register it anywhere.

### Adding a New Test Sequence

To add a new test sequence:

1. **Edit `test_sequences.yaml`** and add your sequence:

```yaml
sequences:
  # Add your new sequence here
  my_custom_sequence:
    name: "My Custom Tests"
    description: "Custom test sequence for specific functionality"
    test_numbers: [20, 21, 23]  # Mix of existing and new tests
    pre_sequence: [1]  # Optional: setup tests
    post_sequence: [16, 17, 18, 19]  # Optional: cleanup tests
```

2. **Use your new sequence**:
```bash
python -m tests.integration.main --sequence my_custom_sequence
```

**That's it!** No code changes needed. Pre/post hooks are automatically executed and displayed in results.

## Best Practices

1. **Error Handling**: Always wrap test logic in try-catch blocks
2. **Logging**: Use `logger` for consistent logging with pretty-printed JSON
3. **Pretty Printing**: Always log request payloads and response data with `json.dumps(data, indent=2)`
4. **Visual Indicators**: Use emojis (🔍, 🤖, 📄, 🗑️, 📋, 📁) for better readability
5. **Verification**: Verify both response status and content
6. **Cleanup**: Ensure tests don't leave persistent state
7. **Documentation**: Add clear descriptions for each test
8. **Incremental Numbers**: Use sequential test numbers (20, 21, 22, etc.)

## Test Selection

The integration test suite supports multiple ways to run tests:

### Predefined Sequences (Recommended)
```bash
# Default: runs basic sequence
python -m tests.integration.main

# Run specific sequences
python -m tests.integration.main --sequence basic
python -m tests.integration.main --sequence optional
python -m tests.integration.main --sequence full

# List available sequences
python -m tests.integration.main --list-sequences
```

### Individual Test Selection
```bash
# List all available tests
python -m tests.integration.main --list-tests

# Run specific test numbers
python -m tests.integration.main --test-numbers 1 3 5

# Run test range
python -m tests.integration.main --test-range 1-5

# Run tests by name
python -m tests.integration.main --test-names "Health Checks" "Create Collections"

# Run tests by partial name matching
python -m tests.integration.main --test-names "Health" "Upload" "Search"

# Mix numbers and names
python -m tests.integration.main --tests 1 "Create Collections" 5

# Exclude specific tests
python -m tests.integration.main --exclude-tests 16
```

## Troubleshooting

### Common Issues

1. **Connection Refused**
   - Ensure containers are running
   - Check server URLs and ports
   - Verify network connectivity

2. **Timeout Errors**
   - Increase `--timeout` value for slow environments
   - Check server performance and resources

3. **Missing Test Files**
   - Verify `--data-dir` path exists
   - Ensure test files are accessible
   - Check file permissions
   - **Recommended default files**: Ensure `multimodal_test.pdf`, `woods_frost.docx`, `table_test.pdf`, and `embedded_table.pdf` are in the data directory
   - When using `--files-with-metadata` or `--files-without-metadata`, verify all specified files exist

4. **Task Failures**
   - Check container logs for detailed error messages
   - Verify Milvus connection and configuration
   - Check model endpoint availability

5. **Collection Name Conflicts**
   - Use `--collection-with-metadata` and `--collection-without-metadata` to avoid conflicts with existing collections
   - The script automatically cleans up test collections via post-sequence hooks, but conflicts may occur if tests are interrupted

6. **Debugging Failed Tests**
   - Post-sequence cleanup tests always run, even if main sequence fails
   - Check the test results table for phase breakdown (🔧 PRE, 📋 MAIN, 🧹 POST)
   - Use `--verbose` for detailed logging during test execution
   - Check container logs for detailed error messages

## Output

- **Console**: Real-time test progress and final results table
- **Log File**: `integration_test.log` with complete test execution details
- **Exit Codes**: `0` for success, `1` for failure
## Local Development Setup

### Prerequisites
- Python 3.12+
- uv (recommended) or pip

### Quick Setup for Local Development

**Option 1: Simple Setup (Recommended for new users)**
```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate     # On Windows

# Install everything at once
uv pip install -r tests/unit/requirements-dev.txt
# or
pip install -r tests/unit/requirements-dev.txt

# Run tests
pytest tests/unit/ -v
```

**Option 2: Step-by-step Setup**
1. **Create and activate a virtual environment:**
   ```bash
   # Using uv (recommended)
   uv venv
   source .venv/bin/activate  # On Linux/Mac
   # or
   .venv\Scripts\activate     # On Windows

   # Or using venv
   python3 -m venv .venv
   source .venv/bin/activate  # On Linux/Mac
   # or
   .venv\Scripts\activate     # On Windows
   ```

2. **Install the package with all optional dependencies:**
   ```bash
   # Using uv
   uv pip install -e .[all]

   # Or using pip
   pip install -e .[all]
   ```

3. **Install test-specific dependencies:**
   ```bash
   # Using uv
   uv pip install -r tests/unit/requirements-test.txt

   # Or using pip
   pip install -r tests/unit/requirements-test.txt
   ```

4. **Run the tests:**
   ```bash
   pytest tests/unit -v
   ```

### Important Notes for Local Development

- **Always install the package with `[all]` optional dependencies first** before installing test requirements
- The test requirements file (`requirements-test.txt`) only contains testing-specific dependencies
- Main package dependencies are managed through `pyproject.toml` and should be installed via `pip install -e .[all]`
- If you encounter "ModuleNotFoundError: No module named 'nvidia_rag'", make sure you've installed the package in editable mode first

### Troubleshooting Local Development

1. **ModuleNotFoundError: No module named 'nvidia_rag'**
   - Solution: Install the package first with `pip install -e .[all]`

2. **Import errors for langchain or opentelemetry packages**
   - Solution: These are included in the `[all]` optional dependencies, make sure to install them

3. **Test failures due to missing dependencies**
   - Solution: Ensure both the package and test requirements are installed in the correct order

4. **MinIO connection errors in tests**
   - Solution: The tests use mocks for external services like MinIO, so these errors should not occur in properly configured tests

### Running Tests with Coverage

```bash
pytest tests/unit -v --cov=src --cov-report=term-missing --cov-report=html:coverage_report
```

### Running Specific Test Files

```bash
# Run specific test file
pytest tests/unit/test_server.py -v

# Run specific test class
pytest tests/unit/test_ingestor_server/test_ingestor_server.py::TestHealthEndpoint -v

# Run specific test method
pytest tests/unit/test_ingestor_server/test_ingestor_server.py::TestHealthEndpoint::test_health_check -v
```
# Linting and Code Quality

This project uses Ruff and Pylint to ensure code quality and consistency.

## Quick Start

### 1. Install Development Dependencies
```bash
pip install -r requirements-dev.txt
```

### 2. Set Up Pre-commit Hooks (Recommended)

Automatic linting and fixing can be run on every commit using pre-commit hooks. This uses the `.pre-commit-config.yaml` in the project root and only runs on files under the `src/` directory.

```bash
pre-commit install
# To analyze and fix all files under src/ manually:
pre-commit run --all-files
```

### 3. Individual Tool Commands (Advanced)
If you want to run tools directly:
```bash
ruff check --fix <file(s)>
ruff format <file(s)>
pylint <file(s)>
```



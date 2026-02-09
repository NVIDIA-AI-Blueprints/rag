# CI Scripts

This directory contains scripts used by the CI/CD pipeline.

## check_markdown_links.py

A comprehensive link checker for all markdown files in the repository.

### Features

- **Internal Link Validation**: Checks that all internal links (relative and absolute paths) point to existing files
- **External Link Validation**: Verifies that external HTTP/HTTPS links are accessible
- **Comprehensive Reporting**: Provides detailed error messages for each broken link
- **Caching**: Avoids duplicate checks of the same external URLs across files
- **Performance**: Uses HEAD requests first for faster external link checking

### Usage

```bash
# Check all markdown files (including external links)
python ci/check_markdown_links.py --root .

# Skip external link checking (faster, for local development)
python ci/check_markdown_links.py --root . --no-external

# Exit immediately on first error (for debugging)
python ci/check_markdown_links.py --root . --fail-fast
```

### Options

- `--root ROOT`: Root directory of the repository (default: current directory)
- `--no-external`: Skip checking external links
- `--fail-fast`: Exit immediately on first error

### CI Integration

This script is automatically run in the GitLab CI pipeline as the `check-markdown-links` job in the test stage.

The job runs:
- On merge requests
- On the default branch
- When triggered manually via web
- On scheduled pipelines

### Exit Codes

- `0`: All links are valid
- `1`: One or more broken links were found or an error occurred

### Dependencies

- Python 3.12+
- `requests` library


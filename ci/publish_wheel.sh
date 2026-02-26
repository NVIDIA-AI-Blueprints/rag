#!/bin/bash
set -e

# Install dependencies
pip install uv==0.8.12
pip install twine

export RELEASE_TYPE=dev

# Handle custom version logic
if [ -n "$CUSTOM_VERSION" ]; then
    echo "Using custom VERSION from CI variable: $CUSTOM_VERSION"
    VERSION=$CUSTOM_VERSION
else
    echo "Using auto-generated version"
    VERSION=$(./ci/get_version.sh)
fi

echo "Using version: $VERSION"
echo "sed -i 's#^version = \".*\"#version = \"$VERSION\"#' pyproject.toml"
sed -i "s#^version = \".*\"#version = \"$VERSION\"#" pyproject.toml

# Handle Artifactory version logic
if [ -n "$ARTIFACTORY_VERSION" ]; then
    echo "Using custom Artifactory version: $ARTIFACTORY_VERSION"
    ARTIFACTORY_VERSION_FINAL=$ARTIFACTORY_VERSION
else
    echo "Using default Artifactory version: 2.5.0.dev"
    ARTIFACTORY_VERSION_FINAL="2.5.0.dev"
fi

# Build first wheel for GitLab Package Registry
uv build
ls -la dist/
export WHEEL_FILE=$(ls dist/*.whl | head -n 1)
echo "Found wheel file: $WHEEL_FILE"

# Publish to GitLab Package Registry
python -m twine upload --repository-url ${CI_API_V4_URL}/projects/${CI_PROJECT_ID}/packages/pypi "$WHEEL_FILE"
echo "Wheel published to GitLab Package Registry successfully"

# Build second wheel for NVIDIA Artifactory with fixed version
echo "Building wheel for Artifactory with version: $ARTIFACTORY_VERSION_FINAL"
sed -i "s#^version = \".*\"#version = \"$ARTIFACTORY_VERSION_FINAL\"#" pyproject.toml
uv build
export ARTIFACTORY_WHEEL_FILE=$(ls dist/*.whl | head -n 1)
echo "Found Artifactory wheel file: $ARTIFACTORY_WHEEL_FILE"

# Publish to NVIDIA Artifactory
python -m twine upload --repository-url https://urm.nvidia.com/artifactory/api/pypi/sw-foundational-rag-dev-pypi-local/ "$ARTIFACTORY_WHEEL_FILE" --username $NVIDIA_ARTIFACTORY_USERNAME --password $NVIDIA_ARTIFACTORY_PASSWORD
echo "Wheel published to NVIDIA Artifactory successfully"
